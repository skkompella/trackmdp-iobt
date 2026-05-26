#!/bin/bash
set -e

SESSION=20260417_100634
FLAC_DIR=iobt_data
NODES_TXT=iobt_data/node_positions.txt
GPS_CSV=iobt_data/${SESSION}_gps2_gps.csv
PARAMS=iobt_data/path_loss_params_${SESSION}.json
OUT_DIR=results/${SESSION}

echo "=== [1/2] Training path-loss params ==="
python collection/flac_to_trackmdp.py --train \
    --flac-dir   "$FLAC_DIR" \
    --nodes-txt  "$NODES_TXT" \
    --gps-csv    "$GPS_CSV" \
    --session    "$SESSION" \
    --step       0.5 \
    --out-params "$PARAMS"

echo "=== [2/2] Generating rankings + transition matrix ==="
python collection/flac_to_trackmdp.py \
    --flac-dir      "$FLAC_DIR" \
    --nodes-csv     iobt_data/nodes_positions.csv \
    --path-loss     "$PARAMS" \
    --infer-session "$SESSION" \
    --out-dir       "$OUT_DIR" \
    --transition \
    --min-dwell 2 \
    --min-prob  0.05