#!/usr/bin/env bash
# Send a message to the Telegram chat.
# Reads bot token + chat id from /etc/strix-halo/telegram.env (mode 0640, root:<user>).
#
# Usage:
#   tg_alert.sh "hello world"
#   tg_alert.sh "<b>bold</b>" --html      (default — supports basic HTML)
#   echo "from stdin" | tg_alert.sh
#
# Exit code is always 0 — alerts are best-effort, must not break callers.

set -uo pipefail

ENV_FILE=/etc/strix-halo/telegram.env
if [[ -r "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
else
    echo "tg_alert: $ENV_FILE not readable, skipping" >&2
    exit 0
fi

if [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]]; then
    echo "tg_alert: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set" >&2
    exit 0
fi

# Accept message from $1 OR stdin
if [[ $# -gt 0 ]]; then
    msg="$1"
else
    msg="$(cat)"
fi

if [[ -z "$msg" ]]; then
    exit 0
fi

# Telegram caps text at 4096 chars per message; truncate with marker.
if (( ${#msg} > 4000 )); then
    msg="${msg:0:3990}…[truncated]"
fi

curl -sS -m 10 -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d "chat_id=${TELEGRAM_CHAT_ID}" \
    -d "parse_mode=HTML" \
    -d "disable_web_page_preview=true" \
    --data-urlencode "text=${msg}" \
    > /dev/null 2>&1 || true
exit 0
