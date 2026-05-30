#!/usr/bin/env bash
set -euo pipefail

cd /home/junyao/code/ANN-CC-bench

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="paper-progress/m3/search-tail-rootcause-2026-05-01/overnight_20260504/codex_${RUN_ID}"
PROMPT="${PROMPT_PATH:-paper-progress/m3/search-tail-rootcause-2026-05-01/overnight_codex_m3_rootcause_prompt_20260504.md}"
SESSION="m3_rootcause_${RUN_ID//[^A-Za-z0-9_]/_}"

mkdir -p "$LOG_DIR"

tmux new-session -d -s "$SESSION" \
  "PROMPT='$PROMPT' LOG_DIR='$LOG_DIR' bash -lc 'cd /home/junyao/code/ANN-CC-bench; PROMPT_TEXT=\"\$(< \"\$PROMPT\")\"; codex exec --cd /home/junyao/code/ANN-CC-bench --sandbox danger-full-access -c \"approval_policy=\\\"never\\\"\" -c \"sandbox_mode=\\\"danger-full-access\\\"\" -o \"\$LOG_DIR/final_message.md\" \"\$PROMPT_TEXT\" > \"\$LOG_DIR/codex_exec.log\" 2>&1 < /dev/null; echo \$? > \"\$LOG_DIR/exit_code\"'"

echo "$SESSION" > "$LOG_DIR/tmux_session"
tmux display-message -p -t "$SESSION" '#{pane_pid}' > "$LOG_DIR/pane_pid"
echo "$LOG_DIR"
