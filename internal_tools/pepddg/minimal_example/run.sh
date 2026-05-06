#!/usr/bin/env bash
# Minimal PepDDG example: scores 10 synthetic mutations and writes output artifacts.
#
# Usage:
#   cd internal_tools/pepddg/minimal_example
#   bash run.sh
#
# Expected output in output/:
#   scored.csv                  - Input + score column + auxiliary columns
#   policy_audit_pepddg.json   - Clean-3 policy check (should pass)
#   gate_metrics_pepddg.json   - Correlation with ddg_exp
#   run_manifest_pepddg.json   - Full config fingerprint
#   run_summary_pepddg.json    - Compact status summary

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Clean previous generated output while preserving output/.gitignore.
mkdir -p "$SCRIPT_DIR/output"
find "$SCRIPT_DIR/output" -mindepth 1 ! -name .gitignore -delete

cd "$REPO_ROOT"
python -m internal_tools.pepddg \
  --config internal_tools/pepddg/minimal_example/config.yaml

echo ""
echo "=== PepDDG minimal example complete ==="
echo "Output files:"
ls -la "$SCRIPT_DIR/output/"
echo ""
echo "Score column preview:"
python -c "
import pandas as pd
df = pd.read_csv('$SCRIPT_DIR/output/scored.csv')
print(df[['target','mutation','rankscore_pepddg_zs','ddg_exp']].to_string(index=False))
"
echo ""
echo "Gate metrics:"
python -c "
import json
with open('$SCRIPT_DIR/output/gate_metrics_pepddg.json') as f:
    m = json.load(f)
print(f'  rho_pred: {m.get(\"rho_pred\", \"N/A\")}')
print(f'  policy:   gate metrics available = {m.get(\"available\", False)}')
"
