#!/bin/bash
#
# Batch process respeaker power recordings to extract transition matrices
# and dominant cycles for each file.
#
# Usage:
#   bash process_respeaker_batch.sh
#   bash process_respeaker_batch.sh --min-dwell 3 --min-prob 0.1
#

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
cd "${PROJECT_ROOT}"

# Collect any command-line args to pass through (e.g. --min-dwell, --min-prob)
EXTRA_ARGS="$@"

echo "=========================================================================="
echo "  BATCH RESPEAKER TRANSITION PROCESSING"
echo "=========================================================================="
echo "  Data directory: ${DATA_DIR}"
echo "  Extra args:    ${EXTRA_ARGS:-none}"
echo ""

# Array of files to process
FILES=(
    "respeaker_power_20260416_132841.npy"
    "respeaker_power_20260416_133649.npy"
    "respeaker_power_20260416_140308.npy"
    "respeaker_power_20260416_145420.npy"
    "respeaker_power_20260416_152738.npy"
    "respeaker_power_20260416_152832.npy"
    "respeaker_power_20260416_153007.npy"
    "respeaker_power_20260416_153953.npy"
    "respeaker_power_20260416_154143.npy"
    "respeaker_power_20260416_154538.npy"
    "respeaker_power_20260416_154736.npy"
    "respeaker_power_20260416_154914.npy"
    "respeaker_power_20260416_155222.npy"
    "respeaker_power_20260416_155935.npy"
    "respeaker_power_20260416_160435.npy"
    "respeaker_power_20260416_160644.npy"
    "respeaker_power_20260416_160748.npy"
    "respeaker_power_20260416_160920.npy"
    "respeaker_power_20260416_161004.npy"
)

TOTAL=${#FILES[@]}
SUCCEEDED=0
FAILED=0

for i in "${!FILES[@]}"; do
    FILE="${FILES[$i]}"
    FILEPATH="${DATA_DIR}/${FILE}"
    NUM=$((i + 1))

    echo "[$NUM/$TOTAL] Processing: ${FILE}"

    if [ ! -f "${FILEPATH}" ]; then
        echo "  ✗ File not found: ${FILEPATH}"
        ((FAILED++))
        echo ""
        continue
    fi

    if python examples/respeaker_transition.py \
        --file "${FILEPATH}" \
        --out-dir "${DATA_DIR}" \
        ${EXTRA_ARGS}; then
        SUCCEEDED=$(( SUCCEEDED + 1 ))
    else
        FAILED=$(( FAILED + 1 ))
    fi

    echo ""
done

echo "=========================================================================="
echo "  BATCH COMPLETE"
echo "=========================================================================="
echo "  Processed: $TOTAL files"
echo "  Succeeded: $SUCCEEDED"
echo "  Failed:    $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "  ✓ All files processed successfully!"
    exit 0
else
    echo "  ✗ ${FAILED} file(s) failed"
    exit 1
fi