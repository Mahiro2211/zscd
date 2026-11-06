#!/bin/bash
# 24GB GPU 最优运行方案(RTX 3090/4090 级)
# - 零样本:窗口全批(batch-windows 16),双实验并行(各占 ~3-4GB)
# - TPT:恢复 TPT 原版 64 视图(top-10% = 6 个置信视图),chunk 32
# 用法: bash scripts/run_24g.sh [zeroshot|tpt|all]
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"          # zscd 仓库根
PROJ_ROOT="${ZSDG_ROOT:-$(dirname "$REPO_DIR")}"   # denseclip/TPT 所在目录
cd "$REPO_DIR"
export ZSDG_ROOT="$PROJ_ROOT"
PY=${PY:-python}   # 如 /opt/conda/bin/python

run_zeroshot_all() {
  for W in openai remoteclip georsclip; do
    $PY -m zscd.run_zeroshot --weights $W --inference slide --stride 112 --batch-windows 16 \
      > /tmp/log_zs_${W}_sem.txt 2>&1 &
    P1=$!
    $PY -m zscd.run_zeroshot --weights $W --baseline notext --inference slide --stride 112 --batch-windows 16 \
      > /tmp/log_zs_${W}_notext.txt 2>&1 &
    P2=$!
    wait $P1 $P2
    echo "==== done $W (semantic + notext) ===="
  done
}

run_tpt_all() {
  for W in openai remoteclip georsclip; do
    $PY -m zscd.run_tpt --weights $W --n-views 64 --tta-steps 1 --lr 5e-3 \
      --chunk 32 --batch-windows 16 > /tmp/log_tpt_${W}.txt 2>&1
    echo "==== done TPT $W ===="
  done
}

case "${1:-all}" in
  zeroshot) run_zeroshot_all ;;
  tpt)      run_tpt_all ;;
  all)      run_zeroshot_all; run_tpt_all ;;
esac
$PY -m zscd.make_report
