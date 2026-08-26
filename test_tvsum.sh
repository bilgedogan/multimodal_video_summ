# CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 0 --dataset tvsum --weights tau \
#         --exp_name exp_4_tvt_diff --diffusion True --fusion_type local_rl --result_dir Summaries/ \
#         --pt_path llama_emb/tvsum_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048
# CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 1 --dataset tvsum --weights tau \
#         --exp_name exp_4_tvt_diff --diffusion True --fusion_type local_rl --result_dir Summaries/ \
#         --pt_path llama_emb/tvsum_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048
# CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 2 --dataset tvsum --weights tau \
#         --exp_name exp_4_tvt_diff --diffusion True --fusion_type local_rl --result_dir Summaries/ \
#         --pt_path llama_emb/tvsum_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048
# CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 3 --dataset tvsum --weights tau \
#         --exp_name exp_4_tvt_diff --diffusion True --fusion_type local_rl --result_dir Summaries/ \
#         --pt_path llama_emb/tvsum_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048
CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 1112 --dataset tvsum --weights tau \
        --exp_name exp_4_diff --diffusion True --fusion_type local_rl --result_dir Summaries/ \
        --pt_path llama_emb/tvsum_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048