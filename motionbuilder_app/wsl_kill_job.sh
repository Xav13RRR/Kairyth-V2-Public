#!/usr/bin/env bash
set -euo pipefail
pid_file="$1"
cancel_file="${pid_file}.cancel"
mkdir -p "$(dirname "$pid_file")"
# Intent persists even when WSL has not started or has not published a PID.
: > "$cancel_file"
if [[ -f "$pid_file" ]]; then
  pid="$(cat "$pid_file")"
  if [[ "$pid" =~ ^[1-9][0-9]*$ ]] && (( pid > 1 )); then
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in {1..30}; do
      if ! kill -0 -- "-$pid" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
    if kill -0 -- "-$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
  fi
  if [[ -f "$pid_file" && "$(cat "$pid_file")" == "$pid" ]]; then
    rm -f "$pid_file"
  fi
fi
