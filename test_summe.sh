CUDA_VISIBLE_DEVICES=0 python test_splits.py --dataset summe --weights tau \
        --exp_name exp_1 --result_dir Summaries/ \
        --pt_path llama_emb/summe_sum/ --num_heads 2 --num_layers 3 --reduced_dim 2048

