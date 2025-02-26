(
CUDA_VISIBLE_DEVICES=0 python select_AMEICL.py --s_n 100 --train_path train_path --test_path test_path --dev_path dev_path >1.txt 2>&1 && \
CUDA_VISIBLE_DEVICES=0 python select_AMEICL_robust.py --s_n 100 --rob_bound 0.1 --train_path train_path --test_path test_path --dev_path dev_path >2.txt 2>&1 \
) & 
