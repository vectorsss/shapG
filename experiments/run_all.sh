#!/bin/bash
# Run every config in configs/ in sequence, logging each to results/logs/.
# Failures are reported but do not stop the batch.

set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TOTAL=0; PASSED=0; FAILED=0; FAILED_CFGS=""

for cfg in configs/*.yaml; do
    name="$(basename "$cfg" .yaml)"
    logfile="${LOG_DIR}/${name}_${TIMESTAMP}.log"
    TOTAL=$((TOTAL + 1))
    echo "============================================================"
    echo "[${TOTAL}] $cfg"
    echo "    log: $logfile"
    if python run_benchmark.py "$cfg" "$@" 2>&1 | tee "$logfile"; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
        FAILED_CFGS="${FAILED_CFGS} ${name}"
    fi
done

echo "============================================================"
echo "Total: ${TOTAL}  Passed: ${PASSED}  Failed: ${FAILED}"
[ -n "$FAILED_CFGS" ] && echo "Failed configs:${FAILED_CFGS}"
exit 0
