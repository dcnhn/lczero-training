#!/usr/bin/env bash
set -euo pipefail

nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits \
| while IFS=',' read -r GPUUUID PID MEM; do
    GPUUUID=$(echo "$GPUUUID" | xargs)
    PID=$(echo "$PID" | xargs)
    MEM=$(echo "$MEM" | xargs)

    USER=$(ps -o user= -p "$PID" 2>/dev/null | xargs || true)
    CMD=$(ps -o comm= -p "$PID" 2>/dev/null | xargs || true)

    echo "$USER  PID=$PID  MEM=${MEM}MiB  CMD=$CMD  GPUUUID=$GPUUUID"
  done
