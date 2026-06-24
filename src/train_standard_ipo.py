import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer

model_id = "AIPlans/Qwen3-0.6b-SFT-hs2"
dataset_id = "Jennny/helpsteer2-helpfulness-preference"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(dataset_id)

# 처음에는 빠른 확인을 위해 subset만 사용
train_dataset = dataset["train"].select(range(512))
eval_dataset = dataset["validation"].select(range(128))

args = DPOConfig(
    output_dir="./outputs/qwen3_standard_ipo_test",
    loss_type="ipo",
    beta=0.01,
    learning_rate=5e-7,

    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,

    max_length=1024,
    max_prompt_length=512,

    bf16=True,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    eval_strategy="steps",
    save_strategy="steps",

    report_to="none",
    remove_unused_columns=False,

    model_init_kwargs={
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    },
    ref_model_init_kwargs={
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    },
)

trainer = DPOTrainer(
    model=model_id,
    ref_model=model_id,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model("./outputs/qwen3_standard_ipo_test/final")
tokenizer.save_pretrained("./outputs/qwen3_standard_ipo_test/final")
