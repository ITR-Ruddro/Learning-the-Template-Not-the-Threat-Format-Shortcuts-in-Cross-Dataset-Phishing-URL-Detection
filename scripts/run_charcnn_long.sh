#!/bin/bash
# E4 re-training with a higher epoch cap (20, early stopping patience 1). Skips finished runs.
cd "$(dirname "$0")/../src"
for mode in canonical hostonly noscheme swapaug; do for seed in 42 43 44; do
  case $mode in orig) name=charCNN;; *) name=charCNN-$mode;; esac
  [ -f ../predictions/charcnn_long/${name}_s${seed}.csv.gz ] && continue
  python step2_charcnn.py $mode $seed 20 charcnn_long > ../logs/charcnn_long_${mode}_s${seed}.log 2>&1 || echo "FAILED $mode $seed"
  tail -1 ../logs/charcnn_long_${mode}_s${seed}.log
done; done
echo ALL_DONE
