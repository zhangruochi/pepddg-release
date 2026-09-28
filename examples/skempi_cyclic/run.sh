#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-${ROOT}/results}"
PLATFORM="${PEPDDG_PLATFORM:-CPU}"
THREADS="${PEPDDG_CPU_THREADS:-4}"

for target in 1SMF 3EQS 3EQY 5XCO; do
  case "$target" in
    1SMF) peptide=I; receptor=F ;;
    3EQS) peptide=B; receptor=A ;;
    3EQY) peptide=C; receptor=A ;;
    5XCO) peptide=B; receptor=A ;;
  esac
  pepddg score-structures \
    --structure "${ROOT}/data/${target}/complex.pdb" \
    --peptide-chain "$peptide" --receptor-chain "$receptor" \
    --mutations "${ROOT}/data/${target}/mutations.csv" \
    --target "$target" --parent-id "${target}_wild_type" \
    --platform "$PLATFORM" --cpu-threads "$THREADS" --n-restarts 7 \
    --output "${OUT}/${target}"
done

python "${ROOT}/report.py" --references "${ROOT}/data/references.csv" --results "$OUT" --output "$OUT/report"
