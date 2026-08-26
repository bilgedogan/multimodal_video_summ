#!/bin/bash
# Eval the 8 diffusion-grid configs from exp_18_summe_diff_grid.sh.
# diffusion / sdedit_t / diffusion_steps MUST match training or the ckpt won't load.

SEED=1112
DATASET=summe
FUSION=local_rl
WEIGHTS=tau    # 'tau' or 'rho'

for W in 0.1 0.5; do
  for T in 5 10; do
    for S in 20 40; do
      WTAG=$(echo $W | tr -d '.')
      EXP="exp_18_w${WTAG}_t${T}_s${S}"
      echo "=== eval ${EXP} ==="
      CUDA_VISIBLE_DEVICES=0 python test_splits.py \
        --seed ${SEED} --dataset ${DATASET} --weights ${WEIGHTS} \
        --exp_name "${EXP}" --fusion_type ${FUSION} \
        --diffusion True --sdedit_t ${T} --diffusion_steps ${S} \
        --result_dir Summaries/ --pt_path llama_emb/summe_sum/ \
        --num_heads 2 --num_layers 3 --reduced_dim 2048
    done
  done
done
