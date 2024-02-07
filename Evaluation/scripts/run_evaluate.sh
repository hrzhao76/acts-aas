#!/bin/bash

DIR="$(dirname "${BASH_SOURCE[0]}")"

for i in {1..4}
do
  "$DIR/evaluate_triton.sh" $i 1 perf_analyzer 100000 "/workspace/acts-aas/Evaluation/results-Feb7"
done
