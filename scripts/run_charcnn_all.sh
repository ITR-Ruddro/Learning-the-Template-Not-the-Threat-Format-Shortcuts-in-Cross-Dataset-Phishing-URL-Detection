#!/usr/bin/env bash
# Trains the 12 char-CNN models (4 input modes x 3 seeds) on CPU. Skips runs whose predictions already exist.
set -u
cd "$(dirname "$0")/../src"
mkdir -p ../logs
for mode in orig noscheme swapaug canonical; do
  for seed in 42 43 44; do
    case $mode in orig) name=charCNN;; *) name=charCNN-$mode;; esac
    [ -f ../predictions/charcnn/${name}_s${seed}.csv.gz ] && { echo "skip $name $seed"; continue; }
    python step2_charcnn.py $mode $seed > ../logs/charcnn_${mode}_s${seed}.log 2>&1 || { echo "FAILED $mode $seed (see logs/)"; exit 1; }
    tail -1 ../logs/charcnn_${mode}_s${seed}.log
  done
done
