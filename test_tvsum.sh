CUDA_VISIBLE_DEVICES=0 python test_splits.py --dataset tvsum --weights tau \
        --exp_name exp_2 --fusion_type global_rl --result_dir Summaries/ \
        --pt_path llama_emb/tvsum_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048