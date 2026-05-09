#!/usr/bin/env python3
"""Standalone out-of-process eval for an LoRA checkpoint.

Loads base model + adapter from a checkpoint dir, runs eval on a JSONL eval
set, and prints eval_loss / eval_token_accuracy. Designed to sidestep the
in-process mid-training eval failures we hit on Strix Halo: a fresh process
gets a clean GPU memory layout, no stale fragmentation, and exits cleanly
when done so the next run/eval gets a clean slate too.

Use when:
  - In-process Trainer eval keeps crashing during training (run with --no-eval
    on the trainer, then call this between runs / at end of training)
  - You want to spot-check a checkpoint mid-run without disturbing it
  - You need a final eval after a training run that was launched --no-eval

Examples:
  python3 scripts/eval_checkpoint.py \\
    --base-model Qwen/Qwen3.5-27B \\
    --adapter /path/to/output/checkpoint-N \\
    --eval-data /path/to/eval.jsonl

  # Use a small subset for fast iteration
  python3 scripts/eval_checkpoint.py --adapter ... --eval-data ... --max-samples 10
"""

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

# Same env as training entry — required before torch import.
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)
os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.6",
)

import psutil
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Out-of-process checkpoint eval")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-27B",
                        help="Base model HF id or local path")
    parser.add_argument("--adapter", required=True,
                        help="Path to LoRA checkpoint dir (contains adapter_model.safetensors)")
    parser.add_argument("--eval-data", required=True,
                        help="Path to eval JSONL")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Cap on eval samples (0 = all)")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--mem-fraction", type=float, default=0.95,
                        help="GPU memory cap (eval-only, no training state)")
    parser.add_argument("--history", default=None,
                        help="Optional JSONL path to append one summary line per run "
                             "(used by train_orchestrator.sh for trend tracking).")
    args = parser.parse_args()

    # Pre-flight: defrag kernel pages so the model load + eval allocs don't
    # immediately hit the same TTM order-0 failure that killed in-process eval.
    try:
        subprocess.run(
            ["sudo", "-n", "/usr/local/bin/gpu-defrag-mem"],
            check=True, timeout=30, capture_output=True,
        )
        print("[pre-load] page compactor + drop_caches done", flush=True)
    except (subprocess.SubprocessError, OSError) as e:
        print(f"[pre-load] defrag helper unavailable ({e}); continuing", flush=True)

    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)

    print(f"Base: {args.base_model}")
    print(f"Adapter: {args.adapter}")
    print(f"Eval data: {args.eval_data}")
    print(f"Max samples: {args.max_samples}, max seq length: {args.max_seq_length}")
    print(f"GPU mem cap: {args.mem_fraction:.2f}")

    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n[2/4] Loading base model + adapter...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    # Switch to evaluation mode (equivalent to nn.Module.eval()) — disables dropout etc.
    model.train(False)
    # Re-enable gradient checkpointing for forward pass — keeps peak attention
    # memory bounded even though we're not training. Same trick as in-process
    # eval, but without the Trainer.evaluation_loop hostility.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    alloc_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  Loaded. GPU alloc: {alloc_gb:.1f}GB", flush=True)

    print("\n[3/4] Loading eval data...")
    records = []
    with open(args.eval_data) as f:
        for line in f:
            rec = json.loads(line)
            msgs = rec.get("messages", [])
            if not msgs:
                continue
            # Same arguments-as-dict normalization as training
            for msg in msgs:
                if "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", {})
                        if isinstance(fn.get("arguments"), str):
                            try:
                                fn["arguments"] = json.loads(fn["arguments"])
                            except (json.JSONDecodeError, TypeError):
                                pass
            try:
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False,
                )
            except Exception:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)[:args.max_seq_length]
            if ids:
                records.append(ids)

    if args.max_samples > 0:
        records = records[:args.max_samples]
    print(f"  Eval samples: {len(records)}")

    print("\n[4/4] Running eval...")
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    n_batches = 0
    with torch.inference_mode():
        for i, ids in enumerate(records):
            input_ids = torch.tensor([ids], device="cuda")
            labels = input_ids.clone()
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss.item()
            total_loss += loss
            n_batches += 1
            # Token accuracy: argmax of logits[:-1] vs labels[1:]
            logits = out.logits[:, :-1, :]
            tgt = labels[:, 1:]
            preds = logits.argmax(dim=-1)
            mask = tgt != -100
            total_tokens += mask.sum().item()
            correct_tokens += ((preds == tgt) & mask).sum().item()
            del out, logits, preds, mask, input_ids, labels, tgt
            if (i + 1) % 5 == 0:
                avg = total_loss / n_batches
                avail = psutil.virtual_memory().available / 1024**3
                print(f"  {i+1}/{len(records)}  avg_loss={avg:.4f}  sys_avail={avail:.1f}GB", flush=True)
            if (i + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    avg_loss = total_loss / max(n_batches, 1)
    token_acc = correct_tokens / max(total_tokens, 1)
    print("\n=== EVAL COMPLETE ===")
    print(f"eval_loss={avg_loss:.4f}")
    print(f"eval_token_accuracy={token_acc:.4f}")
    print(f"eval_samples={n_batches}")
    print(f"eval_tokens={total_tokens}")
    print(f"peak_gpu_memory_gb={torch.cuda.max_memory_allocated()/1024**3:.1f}")

    # Per-checkpoint JSON output for downstream consumers
    out_path = Path(args.adapter) / "eval_metrics.json"
    out_path.write_text(json.dumps({
        "eval_loss": avg_loss,
        "eval_token_accuracy": token_acc,
        "eval_samples": n_batches,
        "eval_tokens": total_tokens,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "base_model": args.base_model,
        "adapter": args.adapter,
    }, indent=2))
    print(f"Metrics: {out_path}")

    # Optional append-only history JSONL for orchestrator trend tracking
    if args.history:
        import math
        from datetime import datetime, timezone
        # Extract step from checkpoint dir name "checkpoint-N" if possible
        step = None
        adapter_basename = Path(args.adapter).name
        if adapter_basename.startswith("checkpoint-"):
            try:
                step = int(adapter_basename.split("-", 1)[1])
            except ValueError:
                pass
        history_entry = {
            "step": step,
            "eval_loss": round(avg_loss, 6),
            "eval_perplexity": round(math.exp(min(avg_loss, 20)), 6),
            "eval_token_accuracy": round(token_acc, 6),
            "n_samples": n_batches,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(args.adapter),
        }
        history_path = Path(args.history)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a") as f:
            f.write(json.dumps(history_entry) + "\n")
        print(f"History: appended to {history_path}")


if __name__ == "__main__":
    main()
