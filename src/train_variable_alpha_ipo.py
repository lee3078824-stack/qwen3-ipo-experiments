import argparse
import json
import math
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer
from trl.trainer.dpo_trainer import entropy_from_logits, selective_log_softmax


class VariableAlphaIPOTrainer(DPOTrainer):
    def __init__(
        self,
        *args,
        alpha_min: float = 0.05,
        alpha_max: float = 0.45,
        alpha_init: float = 0.20,
        alpha_learning_rate: float = 1e-3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not (0.0 < alpha_min < alpha_init < alpha_max < 0.5):
            raise ValueError("alpha must satisfy 0 < alpha_min < alpha_init < alpha_max < 0.5")

        init_position = (alpha_init - alpha_min) / (alpha_max - alpha_min)
        init_raw = math.log(init_position / (1.0 - init_position))

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_learning_rate = alpha_learning_rate
        self.alpha_raw = torch.nn.Parameter(
            torch.tensor(init_raw, dtype=torch.float32, device=self.accelerator.device)
        )
        self._alpha_param_group_added = False

    def current_alpha(self):
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.alpha_raw)

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        if not self._alpha_param_group_added:
            optimizer.add_param_group(
                {
                    "params": [self.alpha_raw],
                    "lr": self.alpha_learning_rate,
                    "weight_decay": 0.0,
                }
            )
            self._alpha_param_group_added = True
        return optimizer

    def _compute_loss(self, model, inputs, return_outputs):
        mode = "train" if self.model.training else "eval"

        non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
        model_kwargs = {k: v for k, v in inputs.items() if k not in non_model_keys}
        model_kwargs["use_cache"] = False
        outputs = model(**model_kwargs)

        input_ids = inputs["input_ids"]
        completion_mask = inputs["completion_mask"]
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_completion_mask = completion_mask[..., 1:].contiguous()

        per_token_logps = selective_log_softmax(shift_logits, shift_labels)
        per_token_logps[shift_completion_mask == 0] = 0.0
        policy_logps = per_token_logps.sum(dim=1)
        policy_chosen_logps, policy_rejected_logps = policy_logps.chunk(2, dim=0)

        if not self.precompute_ref_logps:
            raise ValueError("VariableAlphaIPOTrainer expects precompute_ref_log_probs=True")

        ref_chosen_logps = inputs["ref_chosen_logps"]
        ref_rejected_logps = inputs["ref_rejected_logps"]

        policy_gap = policy_chosen_logps - policy_rejected_logps
        ref_gap = ref_chosen_logps - ref_rejected_logps
        scaled_gap = self.beta * (policy_gap - ref_gap)

        alpha = self.current_alpha().to(dtype=scaled_gap.dtype)
        target_gap = torch.log((1.0 - alpha) / alpha)
        loss = ((scaled_gap - target_gap) ** 2).mean()

        self._log_variable_alpha_metrics(
            mode=mode,
            inputs=inputs,
            shift_logits=shift_logits,
            shift_labels=shift_labels,
            shift_completion_mask=shift_completion_mask,
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            scaled_gap=scaled_gap,
            alpha=alpha,
        )

        return (loss, outputs) if return_outputs else loss

    def _log_variable_alpha_metrics(
        self,
        *,
        mode,
        inputs,
        shift_logits,
        shift_labels,
        shift_completion_mask,
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        scaled_gap,
        alpha,
    ):
        per_token_entropy = entropy_from_logits(shift_logits.detach())
        mask = shift_completion_mask
        entropy_sum = self.accelerator.gather_for_metrics((per_token_entropy * mask).sum()).sum()
        total_tokens = self.accelerator.gather_for_metrics(mask.sum()).sum()
        entropy = (entropy_sum / total_tokens).item() if total_tokens > 0 else 0.0
        self._metrics[mode]["entropy"].append(entropy)

        if mode == "train":
            num_tokens_in_batch = (
                self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            )
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        chosen_logits, rejected_logits = shift_logits.detach().chunk(2, dim=0)
        chosen_mask, rejected_mask = shift_completion_mask.chunk(2, dim=0)
        total_chosen_logits = self.accelerator.gather_for_metrics(
            chosen_logits[chosen_mask.bool()].mean(-1).sum()
        ).sum().item()
        total_chosen_tokens = self.accelerator.gather_for_metrics(chosen_mask.sum()).sum().item()
        total_rejected_logits = self.accelerator.gather_for_metrics(
            rejected_logits[rejected_mask.bool()].mean(-1).sum()
        ).sum().item()
        total_rejected_tokens = self.accelerator.gather_for_metrics(rejected_mask.sum()).sum().item()
        self._metrics[mode]["logits/chosen"].append(
            total_chosen_logits / total_chosen_tokens if total_chosen_tokens > 0 else 0.0
        )
        self._metrics[mode]["logits/rejected"].append(
            total_rejected_logits / total_rejected_tokens if total_rejected_tokens > 0 else 0.0
        )

        predictions = chosen_logits.argmax(dim=-1)
        chosen_labels = shift_labels[: len(shift_labels) // 2]
        correct_predictions = (predictions == chosen_labels) & chosen_mask.bool()
        correct_tokens = self.accelerator.gather_for_metrics(correct_predictions.sum())
        total_tokens = self.accelerator.gather_for_metrics(chosen_mask.sum())
        total_sum = total_tokens.sum()
        accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
        self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps
        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()
        self._metrics[mode]["rewards/chosen"].append(
            self.accelerator.gather(chosen_rewards).mean().item()
        )
        self._metrics[mode]["rewards/rejected"].append(
            self.accelerator.gather(rejected_rewards).mean().item()
        )
        self._metrics[mode]["rewards/accuracies"].append(
            self.accelerator.gather((chosen_rewards > rejected_rewards).float()).mean().item()
        )
        self._metrics[mode]["rewards/margins"].append(
            self.accelerator.gather(chosen_rewards - rejected_rewards).mean().item()
        )
        self._metrics[mode]["logps/chosen"].append(
            self.accelerator.gather(policy_chosen_logps).mean().item()
        )
        self._metrics[mode]["logps/rejected"].append(
            self.accelerator.gather(policy_rejected_logps).mean().item()
        )

        actual_rejected_prob = torch.sigmoid(-scaled_gap.detach())
        alpha_value = alpha.detach().float()
        self._metrics[mode]["alpha"].append(alpha_value.item())
        self._metrics[mode]["target/chosen"].append((1.0 - alpha_value).item())
        self._metrics[mode]["target/rejected"].append(alpha_value.item())
        self._metrics[mode]["prob/chosen"].append(
            self.accelerator.gather(1.0 - actual_rejected_prob).mean().item()
        )
        self._metrics[mode]["prob/rejected"].append(
            self.accelerator.gather(actual_rejected_prob).mean().item()
        )

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        output_dir = output_dir or self.args.output_dir
        if self.args.should_save:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            alpha = self.current_alpha().detach().float().item()
            with open(os.path.join(output_dir, "alpha_config.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "alpha": alpha,
                        "chosen_target": 1.0 - alpha,
                        "rejected_target": alpha,
                        "alpha_min": self.alpha_min,
                        "alpha_max": self.alpha_max,
                        "alpha_raw": self.alpha_raw.detach().float().item(),
                        "alpha_learning_rate": self.alpha_learning_rate,
                    },
                    f,
                    indent=2,
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="AIPlans/Qwen3-0.6b-SFT-hs2")
    parser.add_argument("--dataset_id", type=str, default="Jennny/helpsteer2-helpfulness-preference")
    parser.add_argument("--subset", type=int, default=0, help="0 means use the full train split")
    parser.add_argument("--eval_subset", type=int, default=0, help="0 means use the full validation split")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--alpha_min", type=float, default=0.05)
    parser.add_argument("--alpha_max", type=float, default=0.45)
    parser.add_argument("--alpha_init", type=float, default=0.20)
    parser.add_argument("--alpha_lr", type=float, default=1e-3)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--run_name", type=str, default=None)
    args_cli = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args_cli.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args_cli.dataset_id)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    if args_cli.subset > 0:
        train_dataset = train_dataset.select(range(min(args_cli.subset, len(train_dataset))))
    if args_cli.eval_subset > 0:
        eval_dataset = eval_dataset.select(range(min(args_cli.eval_subset, len(eval_dataset))))

    out_dir = args_cli.output_dir or "./outputs/variable_alpha_ipo"
    run_name = args_cli.run_name or Path(out_dir).name

    training_args = DPOConfig(
        output_dir=out_dir,
        loss_type="ipo",
        beta=args_cli.beta,
        learning_rate=args_cli.lr,
        num_train_epochs=args_cli.epochs,
        per_device_train_batch_size=args_cli.train_batch_size,
        per_device_eval_batch_size=args_cli.eval_batch_size,
        gradient_accumulation_steps=args_cli.gradient_accumulation_steps,
        max_length=args_cli.max_length,
        bf16=True,
        optim="adamw_torch",
        logging_steps=args_cli.logging_steps,
        save_steps=args_cli.save_steps,
        eval_steps=args_cli.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=args_cli.save_total_limit,
        report_to=args_cli.report_to,
        run_name=run_name,
        remove_unused_columns=False,
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=args_cli.eval_batch_size,
        model_init_kwargs={
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        },
    )

    trainer = VariableAlphaIPOTrainer(
        model=args_cli.model_id,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        alpha_min=args_cli.alpha_min,
        alpha_max=args_cli.alpha_max,
        alpha_init=args_cli.alpha_init,
        alpha_learning_rate=args_cli.alpha_lr,
    )

    trainer.train()
    trainer.save_model(f"{out_dir}/final")
    tokenizer.save_pretrained(f"{out_dir}/final")
    print(f"Saved final model, tokenizer, and alpha config to {out_dir}/final")


if __name__ == "__main__":
    main()
