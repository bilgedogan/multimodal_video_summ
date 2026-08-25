CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 0 --dataset summe --weights tau \
        --exp_name 'exp_13' --rl_weight 0.01 --fusion_type local_rl --result_dir Summaries/ \
        --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 1 --dataset summe --weights tau \
       --exp_name 'exp_13' --rl_weight 0.01 --fusion_type local_rl --result_dir Summaries/ \
        --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 2 --dataset summe --weights tau \
        --exp_name 'exp_13' --rl_weight 0.01 --fusion_type local_rl --result_dir Summaries/ \
        --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 3 --dataset summe --weights tau \
        --exp_name 'exp_13' --rl_weight 0.01 --fusion_type local_rl --result_dir Summaries/ \
        --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

# CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 1112 --dataset summe --weights tau \
#         --exp_name 'exp_13' --rl_weight 0.01 --fusion_type local_rl --result_dir Summaries/ \
#         --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 0 --dataset summe --weights tau \
        --exp_name 'exp_14' --rl_weight 0.05 --fusion_type local_rl --result_dir Summaries/ \
        --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 1 --dataset summe --weights tau \
        --exp_name 'exp_14' --rl_weight 0.05  --fusion_type local_rl --result_dir Summaries/ \
        --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 2 --dataset summe --weights tau \
        --exp_name 'exp_14' --rl_weight 0.05 --fusion_type local_rl --result_dir Summaries/ \
        --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 3 --dataset summe --weights tau \
        --exp_name 'exp_14' --rl_weight 0.05 --fusion_type local_rl --result_dir Summaries/ \
        --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

# CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 1112 --dataset summe --weights tau \
#         --exp_name 'exp_14' --rl_weight 0.05 --fusion_type local_rl --result_dir Summaries/ \
#         --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 0 --dataset summe --weights tau \
        --exp_name 'exp_15' --rl_weight 0.5 --fusion_type local_rl --result_dir Summaries/ \
        --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 1 --dataset summe --weights tau \
        --exp_name 'exp_15' --rl_weight 0.5  --fusion_type local_rl --result_dir Summaries/ \
        --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 2 --dataset summe --weights tau \
        --exp_name 'exp_15' --rl_weight 0.5 --fusion_type local_rl --result_dir Summaries/ \
        --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 3 --dataset summe --weights tau \
        --exp_name 'exp_15' --rl_weight 0.5 --fusion_type local_rl --result_dir Summaries/ \
        --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

# CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 1112 --dataset summe --weights tau \
        # --exp_name 'exp_15' --rl_weight 0.5 --fusion_type local_rl --result_dir Summaries/ \
        # --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 0 --dataset summe --weights tau \
        --exp_name 'exp_16' --rl_weight 1 --fusion_type local_rl --result_dir Summaries/ \
        --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 1 --dataset summe --weights tau \
        --exp_name 'exp_16' --rl_weight 1  --fusion_type local_rl --result_dir Summaries/ \
        --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 2 --dataset summe --weights tau \
        --exp_name 'exp_16' --rl_weight 1 --fusion_type local_rl --result_dir Summaries/ \
        --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 3 --dataset summe --weights tau \
        --exp_name 'exp_16' --rl_weight 1 --fusion_type local_rl --result_dir Summaries/ \
        --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

# CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 1112 --dataset summe --weights tau \
#         --exp_name 'exp_16' --rl_weight 1 --fusion_type local_rl --result_dir Summaries/ \
#         --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048
