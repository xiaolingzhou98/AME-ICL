(
CUDA_VISIBLE_DEVICES=0 python evaluate_AMEICL.py --m 4 --l 1 --train_path train_path --test_path test_path --dev_path dev_path --gene_value_path gene_value_path --rob_value_path rob_value_path >1.txt 2>&1 \
) & 
