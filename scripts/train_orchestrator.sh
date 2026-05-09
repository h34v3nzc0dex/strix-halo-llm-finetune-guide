#!/usr/bin/env bash
# Segment Orchestrator - Resume-Safe Train/Eval Loop with Telegram Alerts
#
# Drives (Qwen3.5-27B bf16 LoRA r=128) training as save_steps-aligned
# segments. After each segment, training exits cleanly (max_steps hit), GPU
# memory fully releases, and a separate short-lived process runs eval on the
# just-saved checkpoint. Eval results go to Telegram (delta-vs-prior) + a JSONL
# history. Killing the orchestrator and restarting picks up from the latest
# checkpoint automatically.
#
# Usage:
#   nohup /path/to/scripts/train_orchestrator.sh \
#       --total-steps 448 --save-steps 50 \
#     2>&1 | tee -a /path/to/orchestrator.log &
#
# Smoke test:
#   DRY_RUN=1 /path/to/scripts/train_orchestrator.sh --total-steps 100
#
set -euo pipefail

# ---------- defaults ----------
TOTAL_STEPS=448
SAVE_STEPS=50
OUTPUT_DIR="/path/to/output"
EVAL_DATA="/path/to/eval.jsonl"
HISTORY="/path/to/eval_history.jsonl"
LORA_R=128
LORA_ALPHA=256
EPOCHS=2
GRAD_ACCUM=4
BASE_MODEL="Qwen/Qwen3.5-27B"

SEGMENT_LOG="/path/to/orchestrator_segments.log"
EVAL_LOG="/path/to/orchestrator_evals.log"
TRAIN_LOG="/path/to/training.log"
RUNTIMES_FILE="/path/to/segment_runtimes.txt"

TG_HELPER="/path/to/scripts/tg_alert.sh"
DEFRAG_BIN="/usr/local/bin/gpu-defrag-mem"
VRAM_PATH="/sys/class/drm/card0/device/mem_info_vram_used"
TRAIN_SCRIPT_NAME="train_qwen3_32b.py"

# ---------- arg parse ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --total-steps) TOTAL_STEPS="$2"; shift 2;;
        --save-steps) SAVE_STEPS="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        --eval-data) EVAL_DATA="$2"; shift 2;;
        --history) HISTORY="$2"; shift 2;;
        --lora-r) LORA_R="$2"; shift 2;;
        --lora-alpha) LORA_ALPHA="$2"; shift 2;;
        --epochs) EPOCHS="$2"; shift 2;;
        --grad-accum) GRAD_ACCUM="$2"; shift 2;;
        --base-model) BASE_MODEL="$2"; shift 2;;
        -h|--help)
            grep -E '^# ' "$0" | head -25
            exit 0;;
        *) echo "Unknown arg: $1" >&2; exit 2;;
    esac
done

# ---------- helpers ----------

# Echoes the highest checkpoint-N step in OUTPUT_DIR, or 0 if none.
latest_checkpoint_step() {
    local last
    # shellcheck disable=SC2010 - filenames are controlled
    last=$(ls -d "$OUTPUT_DIR"/checkpoint-*/ 2>/dev/null \
        | sed 's|.*checkpoint-||;s|/||' \
        | sort -n | tail -1 || true)
    if [[ -z "$last" ]]; then
        echo 0
    else
        echo "$last"
    fi
}

# Echo VRAM-used in bytes (0 if path missing).
vram_used_bytes() {
    if [[ -r "$VRAM_PATH" ]]; then
        cat "$VRAM_PATH"
    else
        echo 0
    fi
}

# Wait for: (a) no train process, (b) VRAM <5GB, (c) defrag.
# Returns 0 on success, non-zero on timeout.
wait_gpu_release() {
    local poll=5 cap=120 waited=0
    while pgrep -f "$TRAIN_SCRIPT_NAME" >/dev/null 2>&1; do
        if (( waited >= cap )); then
            echo "[gpu-release] train process still alive after ${cap}s" >&2
            return 1
        fi
        sleep "$poll"; waited=$((waited + poll))
    done
    waited=0
    local threshold=$((5 * 1024 * 1024 * 1024))
    while (( $(vram_used_bytes) >= threshold )); do
        if (( waited >= cap )); then
            echo "[gpu-release] VRAM still >=5GB after ${cap}s ($(vram_used_bytes) bytes)" >&2
            return 2
        fi
        sleep "$poll"; waited=$((waited + poll))
    done
    sync
    if ! sudo -n "$DEFRAG_BIN" >/dev/null 2>&1; then
        echo "[gpu-release] defrag helper failed (continuing)" >&2
    fi
    sleep 10
    return 0
}

# Run one training segment. $1=target_step, $2=is_first (1 or 0).
# Returns the python exit code.
run_segment() {
    local target_step="$1" is_first="$2"
    local resume_flag=""
    if [[ "$is_first" -eq 0 ]]; then
        resume_flag="--resume"
    fi

    local cmd
    cmd="cd /path/to/workspace && source venv/bin/activate && \
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
python3 scripts/$TRAIN_SCRIPT_NAME \
    --bf16-lora --no-eval \
    --output-dir \"$OUTPUT_DIR\" \
    --lora-r \"$LORA_R\" --lora-alpha \"$LORA_ALPHA\" \
    --epochs \"$EPOCHS\" --grad-accum \"$GRAD_ACCUM\" \
    --max-steps \"$target_step\" $resume_flag"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[DRY_RUN] would run: $cmd"
        return 0
    fi

    {
        echo "=== $(date -Is) segment to step $target_step (is_first=$is_first) ==="
        echo "+ $cmd"
    } | tee -a "$SEGMENT_LOG" >> "$TRAIN_LOG"

    set +e
    bash -c "$cmd" 2>&1 | tee -a "$TRAIN_LOG" >> "$SEGMENT_LOG"
    local rc=${PIPESTATUS[0]}
    set -e
    return "$rc"
}

# Run eval on checkpoint-$1. Returns python exit code (NON-FATAL to caller).
run_eval() {
    local step="$1"
    local adapter_dir="$OUTPUT_DIR/checkpoint-$step"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[DRY_RUN] would eval: $adapter_dir against $EVAL_DATA, history=$HISTORY"
        return 0
    fi
    {
        echo "=== $(date -Is) eval step $step ==="
    } >> "$EVAL_LOG"

    set +e
    (
        cd /path/to/workspace
        # shellcheck disable=SC1091
        source venv/bin/activate
        python3 scripts/eval_checkpoint.py \
            --base-model "$BASE_MODEL" \
            --adapter "$adapter_dir" \
            --eval-data "$EVAL_DATA" \
            --max-samples 50 \
            --history "$HISTORY"
    ) >> "$EVAL_LOG" 2>&1
    local rc=$?
    set -e
    return "$rc"
}

# Echo the eval_loss from the last line of the history JSONL (empty if none).
last_eval_loss() {
    [[ -s "$HISTORY" ]] || { echo ""; return; }
    tail -1 "$HISTORY" | jq -r '.eval_loss // empty' 2>/dev/null || echo ""
}

# Echo the eval_loss from the second-to-last line (empty if <2 lines).
prior_eval_loss() {
    [[ -s "$HISTORY" ]] || { echo ""; return; }
    local n
    n=$(wc -l < "$HISTORY")
    if (( n < 2 )); then echo ""; return; fi
    tail -2 "$HISTORY" | head -1 | jq -r '.eval_loss // empty' 2>/dev/null || echo ""
}

# Echo the step from the last line of history (empty if none).
last_history_step() {
    [[ -s "$HISTORY" ]] || { echo ""; return; }
    tail -1 "$HISTORY" | jq -r '.step // empty' 2>/dev/null || echo ""
}

# Format a duration in seconds -> "Hh Mm" or "Mm Ss".
fmt_dur() {
    local s=$1
    if (( s >= 3600 )); then
        printf '%dh %dm' $((s / 3600)) $(((s % 3600) / 60))
    elif (( s >= 60 )); then
        printf '%dm %ds' $((s / 60)) $((s % 60))
    else
        printf '%ds' "$s"
    fi
}

# Average of last 10 entries (float seconds) in $RUNTIMES_FILE, or empty if missing.
# Last-N keeps ETA accurate after a crash+restart that loaded prior segments.
avg_segment_runtime() {
    [[ -s "$RUNTIMES_FILE" ]] || { echo ""; return; }
    tail -n 10 "$RUNTIMES_FILE" | awk '{sum+=$1; n++} END {if (n>0) printf "%.0f", sum/n}'
}

tg_send() {
    local msg="$1"
    if [[ "${DRY_RUN:-0}" == "1" || "${TG_DRY_RUN:-0}" == "1" ]]; then
        echo "[TG_DRY_RUN] $msg"
        return 0
    fi
    "$TG_HELPER" "$msg" || true
}

tg_startup() {
    local step="$1"
    tg_send "$(printf '\xf0\x9f\x9a\x80 <b>training orchestrator started</b>\ncurrent step <code>%s</code>  target <code>%s</code>  segment size <code>%s</code>' \
        "$step" "$TOTAL_STEPS" "$SAVE_STEPS")"
}

# $1=step (just-evaluated), $2=target (==step), $3=eval_loss, $4=prior_loss, $5=segment_runtime_s, $6=eval_runtime_s
tg_segment_success() {
    local step="$1" target="$2" eval_loss="$3" prior_loss="$4" seg_s="$5" eval_s="$6"
    local delta_str=""
    if [[ -n "$prior_loss" && -n "$eval_loss" ]]; then
        local d
        d=$(awk -v a="$eval_loss" -v b="$prior_loss" 'BEGIN{printf "%+0.4f", a - b}')
        local prior_step
        prior_step=$(tail -2 "$HISTORY" | head -1 | jq -r '.step // empty' 2>/dev/null || echo "")
        delta_str=$(printf '  \xce\x94 <code>%s</code> vs step %s' "$d" "$prior_step")
    fi
    local ppl
    ppl=$(awk -v l="$eval_loss" 'BEGIN{printf "%.4f", exp(l)}')
    local eta_str=""
    local avg
    avg=$(avg_segment_runtime)
    if [[ -n "$avg" ]]; then
        local remaining=$(( (TOTAL_STEPS - step + SAVE_STEPS - 1) / SAVE_STEPS ))
        local eta_s=$((avg * remaining))
        eta_str=$(printf '\nETA <code>~%s</code>' "$(fmt_dur "$eta_s")")
    fi
    local seg_fmt eval_fmt
    seg_fmt=$(fmt_dur "$seg_s")
    eval_fmt=$(fmt_dur "$eval_s")
    tg_send "$(printf '\xe2\x9c\x85 <b>segment %s/%s</b>\nloss <code>%s</code>%s\nppl <code>%s</code>\nsegment <code>%s</code>  eval <code>%s</code>%s' \
        "$step" "$TOTAL_STEPS" "$eval_loss" "$delta_str" "$ppl" "$seg_fmt" "$eval_fmt" "$eta_str")"
}

# $1=from_step, $2=target_step, $3=exit_code
tg_segment_failure() {
    local from_step="$1" target="$2" rc="$3"
    local tail_lines=""
    if [[ -s "$SEGMENT_LOG" ]]; then
        tail_lines=$(tail -30 "$SEGMENT_LOG" 2>/dev/null \
            | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g' \
            | tail -c 1500)
    fi
    tg_send "$(printf '\xe2\x9d\x8c <b>segment %s\xe2\x86\x92%s FAILED</b> (exit %s)\n<pre>%s</pre>\nOrchestrator stopped. Investigate before restarting.' \
        "$from_step" "$target" "$rc" "$tail_lines")"
}

# $1=step, $2=exit_code
tg_eval_failure() {
    local step="$1" rc="$2"
    local tail_lines=""
    if [[ -s "$EVAL_LOG" ]]; then
        tail_lines=$(tail -5 "$EVAL_LOG" 2>/dev/null \
            | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g' \
            | tail -c 1500)
    fi
    tg_send "$(printf '\xe2\x9a\xa0\xef\xb8\x8f <b>eval failed at step %s</b> (exit %s)\n<pre>%s</pre>\nContinuing training. History entry skipped.' \
        "$step" "$rc" "$tail_lines")"
}

tg_complete() {
    local final_loss="$1" total_s="$2"
    tg_send "$(printf '\xf0\x9f\x8e\x89 <b>training complete</b>\nfinal eval loss <code>%s</code>\nruntime <code>%s</code>\ncheckpoints: <code>%s</code>' \
        "$final_loss" "$(fmt_dur "$total_s")" "$OUTPUT_DIR")"
}

tg_gpu_release_timeout() {
    local reason="${1:-unknown}"
    tg_send "$(printf '\xe2\x9a\xa0\xef\xb8\x8f <b>GPU release timeout</b>\nreason: %s (after 120s, train script %s).\nOrchestrator stopped. Investigate.' \
        "$reason" "$TRAIN_SCRIPT_NAME")"
}

# Interrupt handler: notify Telegram on SIGINT/SIGTERM.
_orch_interrupt() {
    local sig=$?
    "$TG_HELPER" "$(printf '\xf0\x9f\x9b\x91 <b>training orchestrator interrupted</b>\nReceived signal at step <code>%s</code>.\nRestart with same args to resume.' \
        "$(latest_checkpoint_step)")" || true
    exit 130
}

# Recovery: if last history step lags the latest checkpoint, run the missed eval.
maybe_recover_eval() {
    local cp_step="$1"
    if (( cp_step <= 0 )); then return 0; fi
    local hist_step
    hist_step=$(last_history_step)
    if [[ -z "$hist_step" || "$hist_step" -lt "$cp_step" ]]; then
        echo "[recovery] history step '${hist_step:-none}' < latest checkpoint $cp_step - running missed eval"
        local rc=0
        run_eval "$cp_step" || rc=$?
        if (( rc != 0 )); then
            echo "[recovery] eval failed (rc=$rc) - continuing"
            tg_eval_failure "$cp_step" "$rc"
        else
            echo "[recovery] eval ok"
        fi
    fi
}

# ---------- main ----------
main() {
    trap '_orch_interrupt' INT TERM
    mkdir -p "$(dirname "$HISTORY")" "$(dirname "$RUNTIMES_FILE")"

    local start_step
    start_step=$(latest_checkpoint_step)

    {
        echo "=========================================="
        echo "Orchestrator started: $(date -Is)"
        echo "  total_steps=$TOTAL_STEPS save_steps=$SAVE_STEPS"
        echo "  output_dir=$OUTPUT_DIR"
        echo "  eval_data=$EVAL_DATA"
        echo "  history=$HISTORY"
        echo "  lora_r=$LORA_R alpha=$LORA_ALPHA epochs=$EPOCHS grad_accum=$GRAD_ACCUM"
        echo "  base_model=$BASE_MODEL"
        echo "  starting at step $start_step"
        echo "=========================================="
    } | tee -a "$SEGMENT_LOG"

    tg_startup "$start_step"

    maybe_recover_eval "$start_step"

    local overall_start
    overall_start=$(date +%s)

    local step="$start_step"
    while (( step < TOTAL_STEPS )); do
        # Align target to the next save_steps boundary so the trainer's regular
        # save creates checkpoint-$target (Trainer saves at multiples of save_steps,
        # regardless of where training resumed). For step=87, save_steps=50 → next
        # boundary is 100, not 137.
        local target=$(( ((step / SAVE_STEPS) + 1) * SAVE_STEPS ))
        if (( target > TOTAL_STEPS )); then target=$TOTAL_STEPS; fi
        local is_first=0
        if (( step == 0 )); then is_first=1; fi

        echo "[loop] step=$step target=$target is_first=$is_first" | tee -a "$SEGMENT_LOG"

        local seg_start seg_end seg_s seg_rc=0
        seg_start=$(date +%s)
        run_segment "$target" "$is_first" || seg_rc=$?
        if (( seg_rc != 0 )); then
            tg_segment_failure "$step" "$target" "$seg_rc"
            echo "[FATAL] segment failed (rc=$seg_rc); orchestrator exiting" | tee -a "$SEGMENT_LOG"
            exit 1
        fi
        seg_end=$(date +%s)
        seg_s=$((seg_end - seg_start))
        echo "$seg_s" >> "$RUNTIMES_FILE"

        local gpu_rc=0
        wait_gpu_release || gpu_rc=$?
        if (( gpu_rc != 0 )); then
            local reason
            case "$gpu_rc" in
                1) reason="training process did not exit" ;;
                2) reason="VRAM did not drop below 5GB" ;;
                *) reason="unknown" ;;
            esac
            tg_gpu_release_timeout "$reason"
            echo "[FATAL] GPU release timeout (rc=$gpu_rc, $reason); orchestrator exiting" | tee -a "$SEGMENT_LOG"
            exit 1
        fi

        # Re-discover the actual saved checkpoint. Normally equals $target (because
        # of the alignment fix above), but on the FINAL segment target may be
        # total_steps which isn't a save_steps multiple — trainer saves at the last
        # regular boundary then writes final/. Evaluate the latest real checkpoint.
        local actual_step
        actual_step=$(latest_checkpoint_step)
        local eval_start eval_end eval_s eval_rc=0
        eval_start=$(date +%s)
        if (( actual_step > 0 )); then
            run_eval "$actual_step" || eval_rc=$?
        elif [[ -f "$OUTPUT_DIR/final/adapter_config.json" ]]; then
            # Final segment fallback — trainer wrote final/ but no new checkpoint-N
            echo "[eval] no checkpoint-N, evaluating final/ adapter directly" \
                | tee -a "$SEGMENT_LOG"
            {
                echo "=== $(date -Is) eval final/ (target $target) ==="
            } >> "$EVAL_LOG"
            set +e
            (
                cd /path/to/workspace
                # shellcheck disable=SC1091
                source venv/bin/activate
                python3 scripts/eval_checkpoint.py \
                    --base-model "$BASE_MODEL" \
                    --adapter "$OUTPUT_DIR/final" \
                    --eval-data "$EVAL_DATA" \
                    --max-samples 50 \
                    --history "$HISTORY"
            ) >> "$EVAL_LOG" 2>&1
            eval_rc=$?
            set -e
        else
            echo "[eval] no checkpoint found after segment to step $target — skipping eval" \
                | tee -a "$SEGMENT_LOG"
            eval_rc=99
        fi
        if (( eval_rc != 0 )); then
            tg_eval_failure "${actual_step:-$target}" "$eval_rc"
        else
            eval_end=$(date +%s)
            eval_s=$((eval_end - eval_start))
            local eloss ploss
            eloss=$(last_eval_loss)
            ploss=$(prior_eval_loss)
            tg_segment_success "${actual_step:-$target}" "$target" "$eloss" "$ploss" "$seg_s" "$eval_s"
        fi

        local new_step
        new_step=$(latest_checkpoint_step)
        if (( new_step <= step )); then
            echo "[WARN] checkpoint step did not advance ($step -> $new_step); aborting" | tee -a "$SEGMENT_LOG"
            tg_send "$(printf '\xe2\x9a\xa0\xef\xb8\x8f <b>sanity-check failed</b>: checkpoint did not advance past step %s. Orchestrator stopped.' "$step")"
            exit 1
        fi
        step=$new_step
    done

    local final_loss total_s
    final_loss=$(last_eval_loss)
    total_s=$(( $(date +%s) - overall_start ))
    tg_complete "${final_loss:-unknown}" "$total_s"
    echo "[done] complete at $(date -Is); final_loss=$final_loss runtime=${total_s}s"
}

# Allow `source` for unit-testing helpers without invoking main.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
