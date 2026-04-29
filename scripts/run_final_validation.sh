#!/bin/bash
# QuantumSepsis Shield — Final Validation Script
# Executes all 3 final tasks for publication-ready results
# Run on GPU server: bash scripts/run_final_validation.sh

set -e  # Exit on error

echo "=========================================="
echo "QuantumSepsis Shield — Final Validation"
echo "=========================================="
echo ""

# Ensure we're in the project root
cd ~/QuantumSepsis || { echo "Error: ~/QuantumSepsis not found"; exit 1; }
export PYTHONPATH=.

echo "Step 1/3: Re-running E2E Validation with Fixed Red Team"
echo "--------------------------------------------------------"
python3 scripts/run_e2e_validation.py
echo ""

echo "Step 2/3: Computing Stay-Level Metrics"
echo "---------------------------------------"
python3 scripts/compute_stay_level_metrics.py
echo ""

echo "Step 3/3: Generating Final Results Report"
echo "------------------------------------------"
python3 scripts/generate_final_results.py
echo ""

echo "=========================================="
echo "All tasks complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - data/processed/e2e_validation_results.json"
echo "  - data/processed/stay_level_metrics.json"
echo "  - FINAL_RESULTS.md"
echo ""
echo "Next steps:"
echo "  1. Review FINAL_RESULTS.md"
echo "  2. Test Streamlit dashboard locally: streamlit run scripts/realtime_demo.py"
echo "  3. Commit and push: git add -A && git commit -m 'Final results' && git push"
