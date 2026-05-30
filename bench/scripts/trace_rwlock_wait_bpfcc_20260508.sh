#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  trace_rwlock_wait_bpfcc_20260508.sh <pid>

Environment:
  MIN_US=1000        only print rdlock calls slower than this many us
  OUT=path           write trace output to this file instead of stdout
  STACKS=1           include user/kernel stacks; set STACKS=0 for lower overhead
  LIBC=path          libc path; defaults to /lib/x86_64-linux-gnu/libc.so.6

Example:
  MIN_US=1000 OUT=/tmp/annchor-rdlock.txt \
    bench/scripts/trace_rwlock_wait_bpfcc_20260508.sh 12345
EOF
}

if [[ $# -eq 1 && ( "${1:-}" == "-h" || "${1:-}" == "--help" ) ]]; then
  usage
  exit 0
fi
if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

pid="$1"
min_us="${MIN_US:-1000}"
out="${OUT:-}"
stacks="${STACKS:-1}"
libc="${LIBC:-}"

if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
  echo "pid must be numeric: $pid" >&2
  exit 2
fi

if ! kill -0 "$pid" 2>/dev/null; then
  echo "process is not alive: $pid" >&2
  exit 2
fi

if ! command -v funcslower-bpfcc >/dev/null 2>&1; then
  echo "funcslower-bpfcc is missing; install bpfcc-tools first" >&2
  exit 2
fi

if [[ -z "$libc" ]]; then
  for candidate in /lib/x86_64-linux-gnu/libc.so.6 /usr/lib/x86_64-linux-gnu/libc.so.6; do
    if [[ -r "$candidate" ]]; then
      libc="$candidate"
      break
    fi
  done
fi
if [[ -z "$libc" || ! -r "$libc" ]]; then
  echo "cannot find libc.so.6; set LIBC=/path/to/libc.so.6" >&2
  exit 2
fi

if [[ "$(id -u)" == "0" ]]; then
  cmd=(funcslower-bpfcc -p "$pid" -u "$min_us" -a 1)
else
  cmd=(sudo funcslower-bpfcc -p "$pid" -u "$min_us" -a 1)
fi
if [[ "$stacks" != "0" ]]; then
  cmd+=(-U -K)
fi
cmd+=("$libc:pthread_rwlock_rdlock")

if [[ -n "$out" ]]; then
  mkdir -p "$(dirname "$out")"
  printf 'Tracing PID %s, rdlock wait >= %s us -> %s\n' "$pid" "$min_us" "$out" >&2
  exec "${cmd[@]}" >"$out" 2>&1
else
  printf 'Tracing PID %s, rdlock wait >= %s us\n' "$pid" "$min_us" >&2
  exec "${cmd[@]}"
fi
