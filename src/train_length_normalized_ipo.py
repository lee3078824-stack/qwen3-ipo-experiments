import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path.cwd() / ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path.cwd() / ".hf_cache" / "datasets"))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import DPOConfig
from trl.trainer.dpo_trainer import (
    disable_gradient_checkpointing,
    is_peft_model,
    selective_log_softmax,
    use_adapter,
)

try:
    from train_target_ratio_ipo import TargetRatioIPOTrainer
except ModuleNotFoundError:
    from src.train_target_ratio_ipo import TargetRatioIPOTrainer


class LengthNormalizedTargetRatioIPOTrainer(TargetRatioIPOTrainer):
    def __init__(self, *args, length_norm: str = "mean", objective: str = "target_ratio", **kwargs):
        super().__init__(*args, **kwargs)
        if length_norm not in {"none", "mean", "sqrt"}:
            raise ValueError("length_norm must be one of: none, mean, sqrt")
        if objective not in {"target_ratio", "standard_ipo"}:
            raise ValueError("objective must be one of: target_ratio, standard_ipo")
        self.length_norm = length_norm
        self.objective = objective

    def _normalize_logps(self, logps, lengths):
        if self.length_norm == "none":
            return logps
        return logps / lengths.clamp_min(1).to(logps.dtype).pow(self._length_norm_power())

    def _length_norm_power(self):
        if self.length_norm == "mean":
            return 1.0
        if self.length_norm == "sqrt":
            return 0.5
        return 0.0

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
        completion_lengths = shift_completion_mask.sum(dim=1).clamp_min(1)

        per_token_logps = selective_log_softmax(shift_logits, shift_labels)
        per_token_logps[shift_completion_mask == 0] = 0.0
        policy_logps = self._normalize_logps(per_token_logps.sum(dim=1), completion_lengths)
        policy_chosen_logps, policy_rejected_logps = policy_logps.chunk(2, dim=0)
        chosen_lengths, rejected_lengths = completion_lengths.chunk(2, dim=0)

        if self.precompute_ref_logps:
            ref_chosen_logps = self._normalize_logps(inputs["ref_chosen_logps"], chosen_lengths)
            ref_rejected_logps = self._normalize_logps(inputs["ref_rejected_logps"], rejected_lengths)
        else:
            with torch.no_grad(), disable_gradient_checkpointing(
                self.model,
                self.args.gradient_checkpointing_kwargs,
            ):
                if is_peft_model(model) and self.ref_model is None:
                    unwrapped_model = self.accelerator.unwrap_model(model)
                    with use_adapter(
                        unwrapped_model,
                        adapter_name="ref" if "ref" in unwrapped_model.peft_config else None,
                    ):
                        ref_outputs = self.model(**model_kwargs)
                else:
                    ref_outputs = self.ref_model(**model_kwargs)

            ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
            ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels)
            ref_per_token_logps[shift_completion_mask == 0] = 0.0
            ref_logps = self._normalize_logps(ref_per_token_logps.sum(dim=1), completion_lengths)
            ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)

        policy_logratio = policy_chosen_logps - policy_rejected_logps
        reference_logratio = ref_chosen_logps - ref_rejected_logps
        gap = policy_logratio - reference_logratio

        if self.objective == "target_ratio":
            target = torch.logit(
                torch.tensor(self.target_ratio, device=gap.device, dtype=gap.dtype)
            )
            loss = ((self.beta * gap - target) ** 2).mean()
        else:
            loss = ((gap - 1 / (2 * self.beta)) ** 2).mean()

        self._log_target_ratio_metrics(
            mode=mode,
            inputs=inputs,
            shift_logits=shift_logits,
            shift_labels=shift_labels,
            shift_completion_mask=shift_completion_mask,
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            gap=gap,
        )
        if self.objective == "standard_ipo":
            self._metrics[mode]["target_ratio/target"][-1] = float("nan")
        self._metrics[mode]["length_norm/power"].append(self._length_norm_power())
        self._metrics[mode]["objective/standard_ipo"].append(
            1.0 if self.objective == "standard_ipo" else 0.0
        )
        self._metrics[mode]["completion_length/chosen"].append(
            self.accelerator.gather(chosen_lengths.float()).mean().item()
        )
        self._metrics[mode]["completion_length/rejected"].append(
            self.accelerator.gather(rejected_lengths.float()).mean().item()
        )

        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="AIPlans/Qwen3-0.6b-SFT-hs2")
    parser.add_argument("--dataset_id", type=str, default="Jennny/helpsteer2-helpfulness-preference")
    parser.add_argument("--objective", type=str, choices=["target_ratio", "standard_ipo"], default="target_ratio")
    parser.add_argument("--target_ratio", type=float, default=0.7)
    parser.add_argument("--length_norm", type=str, choices=["none", "mean", "sqrt"], default="mean")
    parser.add_argument("--subset", type=int, default=0, help="0 means use the full train split")
    parser.add_argument("--eval_subset", type=int, default=0, help="0 means use the full validation split")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--precompute_ref_log_probs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args_cli = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args_cli.model_id,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args_cli.dataset_id)

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    if args_cli.subset > 0:
        train_dataset = train_dataset.select(range(min(args_cli.subset, len(train_dataset))))
    if args_cli.eval_subset > 0:
        eval_dataset = eval_dataset.select(range(min(args_cli.eval_subset, len(eval_dataset))))

    if args_cli.objective == "target_ratio":
        default_out_dir = f"./outputs/qwen3_length_normalized_ipo_{args_cli.length_norm}_{args_cli.target_ratio}"
    else:
        default_out_dir = f"./outputs/qwen3_length_normalized_standard_ipo_{args_cli.length_norm}"
    out_dir = args_cli.output_dir or default_out_dir
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
        precompute_ref_log_probs=args_cli.precompute_ref_log_probs,
        precompute_ref_batch_size=args_cli.eval_batch_size,

        model_init_kwargs={
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        },
    )

    trainer = LengthNormalizedTargetRatioIPOTrainer(
        model=args_cli.model_id,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        target_ratio=args_cli.target_ratio,
        length_norm=args_cli.length_norm,
        objective=args_cli.objective,
    )

    trainer.train(resume_from_checkpoint=args_cli.resume_from_checkpoint)
    trainer.save_model(f"{out_dir}/final")
    tokenizer.save_pretrained(f"{out_dir}/final")
    print(f"Saved final model and tokenizer to {out_dir}/final")


if __name__ == "__main__":
    main()
