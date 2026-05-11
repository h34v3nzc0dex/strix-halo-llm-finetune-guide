#!/usr/bin/env python3
"""Out-of-process LoRA-aware eval via `llama-perplexity` (storm-free path).

Replaces `eval_checkpoint.py` on hardware where HF Transformers' mmap-based
weight load triggers the `mm/vmscan.c` TLB-IPI storm — i.e. Strix Halo / Zen 5
mobile boxes that don't expose `INVLPGB` and so fan every batched TLB
shootdown out as a per-CPU IPI to all CPUs in the process's `mm_cpumask`.
On a 27 B-class model this lands at ~340 k IPI/sec average and the system
thrashes itself into a near-freeze around 66 % weight-load.

`llama.cpp`'s mmap+`--mlock` path uses `MAP_PRIVATE` + selective tensor
mlock — completely different memory profile from HF, and storm-free on the
same hardware (measured: ~110 IPI/sec average on identical workload).

Pipeline:
  1. Read up to N samples from eval JSONL.
  2. Apply the HF chat template (matches training tokenization).
  3. Concatenate to a text file.
  4. Run `llama-perplexity` against a base GGUF, optionally with `--lora
     <adapter.gguf>` for runtime LoRA application.
  5. Parse perplexity from stdout.
  6. Write `eval_metrics.json` next to the checkpoint, schema-compatible with
     `eval_checkpoint.py` so the orchestrator's `last_eval_loss` /
     `prior_eval_loss` JSON readers don't need changes.

This script does NOT load PyTorch / FLA / amdgpu directly — only the HF
tokenizer (for the chat template), which is ~50 MB and storm-free.

Quickstart:

  # Eval a base GGUF (merged model) with no LoRA
  python3 scripts/eval_via_llama_perplexity.py \\
      --gguf /path/to/qwen3.5-27b-q8_0.gguf \\
      --eval-data /path/to/eval.jsonl \\
      --metrics-out /tmp/eval_metrics.json

  # Eval a base GGUF + LoRA adapter at runtime (auto-converts safetensors→GGUF on first use)
  python3 scripts/eval_via_llama_perplexity.py \\
      --gguf /path/to/qwen3.5-27b-q8_0.gguf \\
      --adapter /path/to/output/checkpoint-200 \\
      --eval-data /path/to/eval.jsonl \\
      --history /path/to/eval_history.jsonl

Requires:
  - `llama-perplexity` binary built with HIP/ROCm and Qwen3.5 support (build
    >= b867; see Step 6 in the README).
  - `convert_lora_to_gguf.py` from llama.cpp WITH the Qwen3.5 V-head-reorder
    LoRA patch applied (see `patches/`).
  - `transformers` and `huggingface_hub` in your venv (for chat templating).
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime


def _detect_nightly_rocm_lib():
    """Look for the ROCm runtime bundled with the PyTorch nightly wheel.

    On gfx1151 with ROCm 7.1.0, the system libhsa-runtime64.so has a null
    pointer bug; the runtime from the nightly torch wheel (under
    `_rocm_sdk_core/lib`) works. Return the path if found, else None.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        return None
    candidates = []
    for site in sys.path:
        if "site-packages" not in site:
            continue
        candidate = os.path.join(site, "_rocm_sdk_core", "lib")
        if os.path.isdir(candidate):
            candidates.append(candidate)
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser(
        description="Out-of-process LoRA-aware eval via llama-perplexity (storm-free path)",
    )
    parser.add_argument("--gguf", required=True,
                        help="Path to base or merged GGUF (e.g. qwen3.5-27b-q8_0.gguf)")
    parser.add_argument("--lora", default=None,
                        help="Optional path to a GGUF LoRA adapter to apply at runtime. "
                             "If not given but --adapter is, auto-derived to "
                             "<adapter>/gguf/lora-f16.gguf and auto-converted on first use.")
    parser.add_argument("--adapter", default=None,
                        help="Path to an HF/PEFT checkpoint dir (contains adapter_model.safetensors). "
                             "Triggers auto-conversion of the LoRA to GGUF if --lora is not set, "
                             "and auto-derives the step number for --history rows.")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-27B",
                        help="HF model id used only for the chat-template tokenizer")
    parser.add_argument("--eval-data", required=True,
                        help="Path to eval JSONL")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Number of samples to score; 0 = all")
    parser.add_argument("--ctx-size", type=int, default=8192,
                        help="llama-perplexity context size (match training max_seq_length)")
    parser.add_argument("--metrics-out", default=None,
                        help="Where to write eval_metrics.json (defaults to <adapter>/eval_metrics.json "
                             "if --adapter is given, else required)")
    parser.add_argument("--history", default=None,
                        help="Optional JSONL path to append one summary line per run "
                             "(same schema as eval_checkpoint.py, so the orchestrator's readers don't change).")
    parser.add_argument("--text-out", default="/tmp/llama_eval_input.txt",
                        help="Intermediate plain-text file (will be overwritten)")
    parser.add_argument("--llama-perplexity",
                        default="/usr/local/bin/llama-perplexity",
                        help="Path to the llama-perplexity binary")
    parser.add_argument("--lora-converter", default=None,
                        help="Path to convert_lora_to_gguf.py (used for auto-conversion). "
                             "Default: search common locations.")
    parser.add_argument("--rocm-lib-path", default=None,
                        help="Override LD_LIBRARY_PATH with this prepended. Default: auto-detect "
                             "the nightly ROCm runtime bundled in the active venv's torch wheel "
                             "(workaround for ROCm 7.1.0's libhsa null-ptr bug on gfx1151).")
    parser.add_argument("--ngl", type=int, default=999,
                        help="Number of layers to offload to GPU (999 = all)")
    parser.add_argument("--timeout", type=int, default=7200,
                        help="Max seconds for llama-perplexity to run (default 2 h)")
    args = parser.parse_args()

    # Resolve --metrics-out from --adapter if not explicitly given
    if not args.metrics_out:
        if args.adapter:
            args.metrics_out = os.path.join(args.adapter, "eval_metrics.json")
        else:
            sys.exit("ERROR: --metrics-out is required unless --adapter is given")

    # Resolve --lora-converter if not given
    if args.lora_converter is None:
        for candidate in [
            "./llama.cpp/convert_lora_to_gguf.py",
            os.path.expanduser("~/llama.cpp/convert_lora_to_gguf.py"),
            "/opt/llama.cpp/convert_lora_to_gguf.py",
        ]:
            if os.path.isfile(candidate):
                args.lora_converter = candidate
                break

    # Resolve --lora from --adapter if not given; auto-convert if the GGUF
    # doesn't exist yet. Skip if user explicitly provided --lora.
    if not args.lora and args.adapter:
        if not os.path.isdir(args.adapter):
            sys.exit(f"ERROR: --adapter dir not found: {args.adapter}")
        if not os.path.isfile(os.path.join(args.adapter, "adapter_model.safetensors")):
            sys.exit(f"ERROR: --adapter has no adapter_model.safetensors: {args.adapter}")
        lora_dir = os.path.join(args.adapter, "gguf")
        os.makedirs(lora_dir, exist_ok=True)
        args.lora = os.path.join(lora_dir, "lora-f16.gguf")
        if not os.path.isfile(args.lora):
            if not args.lora_converter or not os.path.isfile(args.lora_converter):
                sys.exit("ERROR: --lora-converter not given and convert_lora_to_gguf.py not "
                         "found in default search paths. Build llama.cpp and pass --lora-converter "
                         "or place the patched script at one of: ./llama.cpp/, ~/llama.cpp/, /opt/llama.cpp/")
            print(f"[lora-convert] {args.adapter} → {args.lora}", flush=True)
            conv_cmd = [
                sys.executable, args.lora_converter, args.adapter,
                "--outfile", args.lora, "--outtype", "f16",
            ]
            print(f"  cmd: {' '.join(conv_cmd)}", flush=True)
            conv = subprocess.run(conv_cmd, capture_output=True, text=True, timeout=600)
            if conv.returncode != 0:
                print(f"ERROR: convert_lora_to_gguf failed rc={conv.returncode}", file=sys.stderr)
                print(conv.stderr[-2000:], file=sys.stderr)
                sys.exit(1)
            print(f"[lora-convert] OK ({os.path.getsize(args.lora)/1024/1024:.1f} MiB)",
                  flush=True)
        else:
            print(f"[lora-convert] cached: {args.lora}", flush=True)

    if not os.path.isfile(args.gguf):
        sys.exit(f"ERROR: GGUF not found: {args.gguf}")
    if not os.path.isfile(args.llama_perplexity):
        sys.exit(f"ERROR: llama-perplexity binary not found: {args.llama_perplexity}")
    if not os.path.isfile(args.eval_data):
        sys.exit(f"ERROR: eval data not found: {args.eval_data}")

    # ----- Step 1+2: read JSONL + apply chat template -----
    print(f"[1/3] Loading tokenizer from {args.base_model}", flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print(f"[2/3] Reading + templating samples from {args.eval_data}", flush=True)
    texts = []
    skipped = 0
    with open(args.eval_data) as f:
        for line in f:
            if args.max_samples > 0 and len(texts) >= args.max_samples:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            msgs = rec.get("messages", [])
            if not msgs:
                skipped += 1
                continue
            # Normalize tool-call arguments from JSON-as-string to dict
            # (same shape eval_checkpoint.py expects)
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
            except Exception as e:
                print(f"  skipped sample (template error: {e})", flush=True)
                skipped += 1
                continue
            texts.append(text)

    if not texts:
        sys.exit("ERROR: no samples produced text after template application")

    # Join with a doc-separator newline so llama-perplexity treats them as
    # distinct contexts but still computes joint PPL.
    full_text = "\n\n".join(texts)
    with open(args.text_out, "w") as f:
        f.write(full_text)
    n_chars = len(full_text)
    n_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))
    print(f"  Samples used: {len(texts)} (skipped {skipped})", flush=True)
    print(f"  Wrote {n_chars} chars / ~{n_tokens} tokens to {args.text_out}", flush=True)

    # llama-perplexity needs at least 2*ctx tokens; shrink ctx if needed.
    min_tokens_for_ctx = 2 * args.ctx_size
    eff_ctx = args.ctx_size
    if n_tokens < min_tokens_for_ctx:
        # halve to nearest power of 2 above min usable size (512)
        eff_ctx = max(512, 2 ** int(math.log2(n_tokens // 2)))
        print(f"  WARNING: only {n_tokens} tokens; reducing -c {args.ctx_size}→{eff_ctx}",
              flush=True)

    # ----- Step 3: run llama-perplexity -----
    print(f"[3/3] Running llama-perplexity (ctx={eff_ctx}, ngl={args.ngl})", flush=True)
    env = os.environ.copy()
    rocm_lib = args.rocm_lib_path or _detect_nightly_rocm_lib()
    if rocm_lib:
        # On gfx1151, ROCm 7.1.0's libhsa-runtime64.so has a null-pointer bug.
        # Prepend the nightly torch-wheel's bundled HSA runtime so it wins the
        # dlopen race. Harmless on other archs.
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{rocm_lib}:/opt/rocm-7.1.0/lib"
        if existing:
            env["LD_LIBRARY_PATH"] += f":{existing}"
        print(f"  LD_LIBRARY_PATH prefix: {rocm_lib}", flush=True)

    cmd = [
        args.llama_perplexity,
        "-m", args.gguf,
        "-f", args.text_out,
        "-ngl", str(args.ngl),
        "--mlock",
        "-c", str(eff_ctx),
        "--no-warmup",
    ]
    if args.lora:
        if not os.path.isfile(args.lora):
            sys.exit(f"ERROR: --lora path not found: {args.lora}")
        cmd += ["--lora", args.lora]
        print(f"  with --lora {args.lora}", flush=True)
    print(f"  cmd: {' '.join(cmd)}", flush=True)
    t0 = datetime.now()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=args.timeout)
    t1 = datetime.now()
    runtime_sec = (t1 - t0).total_seconds()

    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(f"ERROR: llama-perplexity exited with code {proc.returncode}", file=sys.stderr)
        print("--- tail of output ---", file=sys.stderr)
        print(out[-2000:], file=sys.stderr)
        sys.exit(1)

    # Parse perplexity. Formats seen:
    #   "Final estimate: PPL = 7.5023 +/- 0.12345"
    #   "[1]X.XXXX,[2]Y.YYYY,..."  (per-chunk; we want the last one if no Final)
    final_match = re.search(
        r"Final estimate:?\s*PPL\s*=\s*([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)",
        out,
    )
    if final_match:
        ppl = float(final_match.group(1))
        ppl_source = "Final estimate"
    else:
        # Fallback: last per-chunk number that looks like a PPL
        chunk_matches = re.findall(r"\[\d+\]\s*([0-9]+\.[0-9]+)", out)
        if chunk_matches:
            ppl = float(chunk_matches[-1])
            ppl_source = f"last of {len(chunk_matches)} chunk PPLs"
        else:
            print("ERROR: no perplexity number found in output", file=sys.stderr)
            print("--- tail of output ---", file=sys.stderr)
            print(out[-2000:], file=sys.stderr)
            sys.exit(1)

    eval_loss = math.log(ppl)

    metrics = {
        "eval_loss": eval_loss,
        "perplexity": ppl,
        "perplexity_source": ppl_source,
        "n_samples": len(texts),
        "n_tokens": n_tokens,
        "ctx_size": eff_ctx,
        "method": "llama-perplexity",
        "gguf_path": args.gguf,
        "lora_path": args.lora,
        "runtime_sec": runtime_sec,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    # Per-checkpoint metrics file (always written)
    os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    # Optional append-only history JSONL for orchestrator trend tracking.
    # Schema matches eval_checkpoint.py so the orchestrator's last_eval_loss /
    # prior_eval_loss readers don't need changes.
    if args.history:
        from datetime import timezone
        step = None
        if args.adapter:
            adapter_basename = os.path.basename(args.adapter.rstrip("/"))
            if adapter_basename.startswith("checkpoint-"):
                try:
                    step = int(adapter_basename.split("-", 1)[1])
                except ValueError:
                    pass
        history_entry = {
            "step": step,
            "eval_loss": round(eval_loss, 6),
            "eval_perplexity": round(ppl, 6),
            "eval_token_accuracy": None,  # not computed by llama-perplexity
            "n_samples": len(texts),
            "n_tokens": n_tokens,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checkpoint": args.adapter or args.gguf,
            "method": "llama-perplexity",
        }
        hist_path = args.history
        os.makedirs(os.path.dirname(hist_path) or ".", exist_ok=True)
        with open(hist_path, "a") as f:
            f.write(json.dumps(history_entry) + "\n")
        print(f"[history] appended to {hist_path}", flush=True)

    print()
    print("=== eval complete ===")
    print(f"  perplexity:   {ppl:.4f}  ({ppl_source})")
    print(f"  eval_loss:    {eval_loss:.4f}")
    print(f"  n_samples:    {len(texts)}")
    print(f"  n_tokens:     {n_tokens}")
    print(f"  ctx_size:     {eff_ctx}")
    print(f"  runtime:      {runtime_sec:.1f} s")
    print(f"  metrics file: {args.metrics_out}")
    print(f"=== EVAL COMPLETE ===  (sentinel for orchestrator/dashboard log parsers)")


if __name__ == "__main__":
    main()
