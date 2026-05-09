#!/usr/bin/env python3
"""Minimal training-script skeleton compatible with train_orchestrator.sh.

This is the smallest script that satisfies the orchestrator's contract
(see README "Training script — the contract"). It uses TRL's SFTTrainer
on a HuggingFace JSONL dataset of {"messages": [...]} chat conversations
with bf16 LoRA. Domain-specific bits (your dataset, tool definitions,
chat template, callbacks) are NOT included — add what you need.

Tested against:
    transformers 5.4.0
    trl 0.29.1
    peft 0.18.1
    accelerate 1.13.0
    bitsandbytes built from source for ROCm gfx1151
    PyTorch 2.11.0+rocm7.13 nightly

Usage (called by train_orchestrator.sh, so all args must be supported):

    python3 train.py \\
        --base-model Qwen/Qwen3.5-27B \\
        --train-data /path/to/train.jsonl \\
        --output-dir /path/to/output \\
        --bf16-lora --no-eval \\
        --lora-r 128 --lora-alpha 256 \\
        --epochs 2 --grad-accum 4 \\
        --max-steps 100 --resume

Stand-alone (no orchestrator):

    python3 train.py \\
        --base-model Qwen/Qwen3-0.6B \\
        --train-data /path/to/train.jsonl \\
        --output-dir ./output \\
        --bf16-lora --epochs 1
"""
import argparse
import os

# CRITICAL — set before importing torch
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)
os.environ["PYTORCH_ALLOC_CONF"] = (
    "expandable_segments:True,garbage_collection_threshold:0.6"
)

import json
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_jsonl_dataset(path, tokenizer, max_seq_length):
    """Load {"messages": [...]} JSONL, render via chat template, tokenize."""
    records = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            msgs = rec.get("messages", [])
            if not msgs:
                continue
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
            )
            ids = tokenizer.encode(text, add_special_tokens=False)[:max_seq_length]
            if ids:
                records.append({"input_ids": ids})
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=8192)

    # Orchestrator-required flags
    parser.add_argument("--bf16-lora", action="store_true",
                        help="Train with bf16 weights + LoRA (no quantization).")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Overrides --epochs when > 0. Used by orchestrator.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-steps", type=int, default=50,
                        help="Must match orchestrator's --save-steps.")
    args = parser.parse_args()

    torch.cuda.set_per_process_memory_fraction(0.80)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}, HIP: {torch.version.hip}")

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Data
    train_dataset = load_jsonl_dataset(
        args.train_data, tokenizer, args.max_seq_length,
    )
    print(f"Train samples: {len(train_dataset)}")

    # 3. Model — eager attention is required for Qwen3.5 hybrid GatedDeltaNet
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # 4. LoRA config — adapt target_modules to your model architecture
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # 5. SFTConfig — note max_steps overrides epochs when > 0
    config = SFTConfig(
        output_dir=args.output_dir,
        push_to_hub=False,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=5e-5,
        warmup_steps=30,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_length=args.max_seq_length,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",  # requires bnb-from-source build
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=5,
        eval_strategy="no",  # orchestrator handles eval out-of-process
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to="none",
    )

    # 6. Train
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=config,
        peft_config=peft_config,
    )

    if args.resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # 7. Final save
    final_dir = f"{args.output_dir}/final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final adapter to {final_dir}")


if __name__ == "__main__":
    main()
