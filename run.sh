#!/bin/bash
# 统一运行脚本：用 isaacsim_5.1 的 kit python 运行 InstinctLab 训练/回放/导出。
# 用法示例：
#   ./run.sh train --task=Instinct-Locomotion-Flat-D1H-v0 --headless --num_envs=64 --max_iterations=10
#   ./run.sh play  --task=Instinct-Locomotion-Flat-D1H-Play-v0 --exportonnx --load_run=<run> --num_envs=1
set -e
ISAAC_PY=/home/you/isaacsim_5.1/python.sh
cd "$(dirname "$0")"

cmd="$1"; shift || true
case "$cmd" in
  train) exec "$ISAAC_PY" scripts/instinct_rl/train.py "$@" ;;
  play)  exec "$ISAAC_PY" scripts/instinct_rl/play.py "$@" ;;
  plot)  exec "$ISAAC_PY" scripts/instinct_rl/plotter.py "$@" ;;
  python) exec "$ISAAC_PY" "$@" ;;
  *) echo "用法: $0 {train|play|plot|python} [args...]" >&2; exit 1 ;;
esac
