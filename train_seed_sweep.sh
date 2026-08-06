#!/bin/bash
# Train + eval all 5 splits across seeds 1112,1,2,3,4 for comparison.
# Usage: ./train_seed_sweep.sh <dataset> <exp_name> <fusion_type>
DATASET=${1:-summe}
EXP_NAME=${2:-exp_6}
FUSION_TYPE=${3:-global_rl}
PT_PATH=${4:-llama_emb/summe_sum/}
SEEDS=(0 1 2 3 1112)


for SEED in "${SEEDS[@]}"; do
    for SPLIT in 0 1 2 3 4; do
        CUDA_VISIBLE_DEVICES=0 python train.py --dataset "$DATASET" --split_idx "$SPLIT" \
            --tag "${DATASET}_split${SPLIT}" --seed "$SEED" --exp_name "$EXP_NAME" \
            --fusion_type "$FUSION_TYPE" --pt_path  "$PT_PATH" --diversity_weight 0.1
    done
    # python test_splits.py --dataset "$DATASET" --weights rho --exp_name "$EXP_NAME" \
    #     --seed "$SEED" --fusion_type "$FUSION_TYPE"
    python test_splits.py --dataset "$DATASET" --weights tau --exp_name "$EXP_NAME" \
        --seed "$SEED" --fusion_type "$FUSION_TYPE" --pt_path "$PT_PATH"
done
