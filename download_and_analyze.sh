#!/usr/bin/env bash
# download_and_analyze.sh
# Run after all Modal jobs complete to pull results and run analysis.
#
# Usage (from project root, with PYTHONIOENCODING=utf-8):
#   bash download_and_analyze.sh

set -e
export PYTHONIOENCODING=utf-8

echo "=== Downloading long-context results (vLLM-patched) ==="
python -m modal volume get spec-dec-m5-results eagle3_longctx_patched/ results/eagle3_longctx_patched/ || echo "[skip: not found yet]"

echo "=== Downloading long-context results (SGLang) ==="
python -m modal volume get spec-dec-m5-results eagle3_longctx_sglang/ results/eagle3_longctx_sglang/ || echo "[skip: not found yet]"

echo "=== Downloading full hidden-state parquets ==="
python -m modal volume get spec-dec-m5-results eagle3_hidden_full/ results/eagle3_hidden_full/ || echo "[skip: not found yet]"

echo ""
echo "=== Running long-context analysis ==="
python analyze_eagle3_longctx_full.py || echo "[skip: no results yet]"

echo ""
echo "=== Running full hidden-state probe ==="
python analyze_perstep_hidden_full.py || echo "[skip: no parquets yet]"

echo ""
echo "Done. Check results/eagle3_longctx_full/ and results/perstep_signal/hidden_full_audit.md"
