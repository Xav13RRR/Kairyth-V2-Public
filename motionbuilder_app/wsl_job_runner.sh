#!/usr/bin/env bash
set -euo pipefail
pid_file="$1"
shift
cancel_file="${pid_file}.cancel"
mkdir -p "$(dirname "$pid_file")"
child=""
cleanup() {
  trap - TERM INT EXIT
  # $! also covers a signal arriving between the background launch and child=$!.
  local group="${child:-${!:-}}"
  if [[ "$group" =~ ^[1-9][0-9]*$ ]] && kill -0 -- "-$group" 2>/dev/null; then
    kill -TERM -- "-$group" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-$group" 2>/dev/null || true
  fi
  if [[ -f "$pid_file" && "$(cat "$pid_file")" == "$group" ]]; then
    rm -f "$pid_file"
  fi
  # Never remove cancel_file: all stages/late WSL startups of this terminal job
  # share it. New jobs have separate UUID directories, so do not inherit it.
}
trap 'exit 130' TERM INT
trap cleanup EXIT
# First check covers a WSL startup that arrives after Windows cancellation.
if [[ -e "$cancel_file" ]]; then
  exit 130
fi
setsid "$@" &
child="$!"
printf '%s' "$child" > "$pid_file"
# Second check closes the cancellation-during-launch / PID-publication window.
if [[ -e "$cancel_file" ]]; then
  exit 130
fi
set +e
wait "$child"
code="$?"
set -e
# EXIT cleanup also kills surviving descendants if the group leader exited.
exit "$code"
