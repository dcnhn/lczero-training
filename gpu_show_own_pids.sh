#!/usr/bin/env bash
set -euo pipefail

USER_TO_KILL="dnnguyen"
WAIT_SECONDS=3

# Collect unique GPU PIDs from nvidia-smi, then filter by owner == USER_TO_KILL
GPU_PIDS=$(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
  | awk '{print $1}' \
  | sort -n | uniq \
  | while read -r pid; do
      [[ -z "$pid" ]] && continue
      owner=$(ps -o user= -p "$pid" 2>/dev/null | awk '{print $1}')
      if [[ "$owner" == "$USER_TO_KILL" ]]; then
        echo "$pid"
      fi
    done \
  | sort -n | uniq
)

if [[ -z "${GPU_PIDS}" ]]; then
  echo "No GPU processes found for user: $USER_TO_KILL"
  exit 0
else
  echo "Found GPU processes for user: $USER_TO_KILL"
  for pid in ${GPU_PIDS}; do
    cmd=$(ps -o comm= -p "$pid" 2>/dev/null | xargs || true)
    echo "  USER=$USER_TO_KILL  PID=$pid  CMD=$cmd"
  done
fi

