# CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 0 --dataset summe --weights tau \
#         --exp_name 'exp_4_tvt_diff' --tr_val_ts True --diffusion True --fusion_type local_rl --result_dir Summaries/ \
#         --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

# CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 1 --dataset summe --weights tau \
#        --exp_name 'exp_4_tvt_diff' --tr_val_ts True --diffusion True --fusion_type local_rl --result_dir Summaries/ \
#         --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

# CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 2 --dataset summe --weights tau \
#         --exp_name 'exp_4_tvt_diff' --tr_val_ts True --diffusion True --fusion_type local_rl --result_dir Summaries/ \
#         --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

# CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 3 --dataset summe --weights tau \
#         --exp_name 'exp_4_tvt_diff' --tr_val_ts True --diffusion True --fusion_type local_rl --result_dir Summaries/ \
#         --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 1112 --dataset summe --weights tau \
        --exp_name 'exp_18_w05_t10_s20' --sdedit_t 10 --diffusion_steps 20 --diffusion True --fusion_type local_rl --result_dir Summaries/ \
        --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 1112 --dataset summe --weights tau \
        --exp_name 'exp_18_w05_t10_s40' --sdedit_t 10 --diffusion_steps 40 --diffusion True --fusion_type local_rl --result_dir Summaries/ \
        --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048
