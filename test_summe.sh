CUDA_VISIBLE_DEVICES=0 python test_splits.py --seed 0 --dataset summe --weights tau \
        --exp_name exp_4 --fusion_type local_rl --result_dir Summaries/ \
        --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

