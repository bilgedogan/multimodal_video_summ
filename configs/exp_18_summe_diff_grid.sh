#!/bin/bash
# Diffusion ablation grid on SumMe, seed 1112, fusion_type=local_rl
#   diffusion_weight : 0.1 , 0.5
#   sdedit_t         : 5   , 10
#   diffusion_steps  : 20  , 40
# 8 configs x 5 splits = 40 runs.
# exp_name pattern: exp_18_w<weight>_t<sdedit_t>_s<steps>
# Results: Summaries/exp_18_w01_t5_s20/seed_1112/summe/summe_split<i>/

SEED=1112
DATASET=summe
FUSION=local_rl

for W in 0.1 0.5; do
  for T in 1; do
    for S in 20 40; do
      WTAG=$(echo $W | tr -d '.')
      EXP="exp_18_w${WTAG}_t${T}_s${S}"
      echo "=== ${EXP} (diffusion_weight=${W} sdedit_t=${T} diffusion_steps=${S}) ==="
      for SPLIT in 0 1 2 3 4; do
        CUDA_VISIBLE_DEVICES=0 python train.py \
          --seed ${SEED} --exp_name "${EXP}" \
          --diffusion True --diffusion_weight ${W} --sdedit_t ${T} --diffusion_steps ${S} \
          --fusion_type "${FUSION}" --tag ${DATASET}_split${SPLIT} --split_idx ${SPLIT} \
          --model summe_head2_layer3 --lr 0.000119 --epochs 200 --dataset ${DATASET} \
          --reduced_dim 2048 --num_heads 2 --num_layers 3 \
          --pt_path 'llama_emb/summe_sum/'
      done
    done
  done
done
