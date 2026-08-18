#!/bin/bash
# Significance testing over Summaries/.
#
#   1. paired t-test across seeds, all experiment pairs   -> ttest_seeds.py
#   2. Wilcoxon signed-rank + Holm-Bonferroni, in groups  -> wilcoxon_test.py
#
# Wilcoxon groups (each group is its own Holm family, per dataset x metric):
#   g1_baseline   exp_0            vs exp_1 .. exp_10
#   g2_exp1_exp2  exp_1            vs exp_2 + exp_2_diff*
#   g3_exp2_diff  exp_2            vs exp_2_diff*
#   g4_exp3_exp4  exp_3            vs exp_4 + exp_4_diff*
#   g5_exp4_diff  exp_4            vs exp_4_diff*
#
# Experiments with no results directory are reported and skipped.
#
# Usage:
#   bash statistical_test.sh [weights] [granularity]
#     weights      tau (default) | rho   -- which checkpoint-selection results to read
#     granularity  split (default, n=25 (seed,split) pairs) | seed (n=5 per-seed means)
set -u

WEIGHTS=${1:-tau}
GRANULARITY=${2:-split}
RESULT_DIR=Summaries
OUT_DIR=${RESULT_DIR}/stats

BASELINE_EXPS=exp_1,exp_2,exp_3,exp_4,exp_5,exp_6,exp_7,exp_8,exp_9,exp_10
EXP2_DIFF=exp_2_diff,exp_2_diff_2,exp_2_diff_3,exp_2_diff_4
EXP4_DIFF=exp_4_diff,exp_4_diff_2,exp_4_diff_3,exp_4_diff_4

echo "=============================================================="
echo " Paired t-test across seeds (all pairs) — weights=${WEIGHTS}"
echo "=============================================================="
python ttest_seeds.py --result_dir "${RESULT_DIR}" --weights "${WEIGHTS}"

run_wilcoxon () {
    local tag=$1 baseline=$2 exps=$3
    echo
    echo "=============================================================="
    echo " Wilcoxon ${tag}: ${baseline} vs ${exps}"
    echo "=============================================================="
    python wilcoxon_test.py \
        --result_dir "${RESULT_DIR}" \
        --out_dir "${OUT_DIR}" \
        --weights "${WEIGHTS}" \
        --granularity "${GRANULARITY}" \
        --baseline "${baseline}" \
        --exps "${exps}" \
        --tag "${tag}"
}

run_wilcoxon g1_baseline  exp_0 "${BASELINE_EXPS}"
run_wilcoxon g1_baseline_globdiff exp_0 "${EXP2_DIFF}"
run_wilcoxon g1_baseline_locdiff exp_0 "${EXP4_DIFF}"
run_wilcoxon g2_exp1_exp2 exp_1 "exp_2,${EXP2_DIFF}"
run_wilcoxon g3_exp2_diff exp_2 "${EXP2_DIFF}"
run_wilcoxon g4_exp3_exp4 exp_3 "exp_4,${EXP4_DIFF}"
run_wilcoxon g5_exp4_diff exp_4 "${EXP4_DIFF}"

echo
echo "Reports written to ${OUT_DIR}/"
