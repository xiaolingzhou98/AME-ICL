import os
import itertools
import argparse
import pickle as pkl
import random
import torch
import math
import json
import string
import logging
import numpy as np
import pdb
import pprint
from tqdm import tqdm
from collections import Counter, defaultdict

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import GPT2Tokenizer, AutoTokenizer

from metaicl.model_evaluate_robust import MetaICLModel # using the robust evalue code
from metaicl.data import MetaICLData
from utils.data import load_data, random_subset_of_comb
from config.config import OUT_SELECT
import pandas as pd




def main(logger, args):

    if args.gpt2.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2, cache_dir="cached")
    elif "gpt-j" in args.gpt2:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b", cache_dir="cached")
    elif "gpt-neo-" in args.gpt2:
        tokenizer = GPT2Tokenizer.from_pretrained(f"EleutherAI/{args.gpt2}", cache_dir="cached")
    elif "opt" in args.gpt2:
        tokenizer = GPT2Tokenizer.from_pretrained(f"facebook/{args.gpt2}", cache_dir="cached")

    test_data = pd.read_json(args.train_path, lines=True)
    train_data = pd.read_json(args.test_path, lines=True)
    dev_data = pd.read_json(args.dev_path, lines=True)


    values1 = np.load(args.gene_value_path)
    values2 = np.load(args.rob_value_path)

    total_value = values1 + args.l * values2

    max_indices = np.argsort(-total_value)

    selected_samples = []
    zeros_list = [0] * args.num_class
    assert args.m % args.num_class == 0
    for i in (max_indices):
        for j in range(args.num_class):
            if (train_data.iloc[i]["output"] == dev_data.iloc[0]["options"][j]):
                zeros_list[j] = zeros_list[j] + 1 
            if (zeros_list[j]<=args.m/args.num_class and train_data.iloc[i]["output"] == dev_data.iloc[0]["options"][j]):
                selected_samples.append(train_data.iloc[i].copy())
        if (len(selected_samples)>=args.m):
            break        

    ### checkpoint ...
    checkpoint = args.checkpoint
    metaicl_model = MetaICLModel(logger)

    metaicl_model.load(checkpoint, gpt2=args.gpt2)
    metaicl_model.to_device()
    metaicl_model.eval()

    # setup hyperparams for data
    max_length_per_example = args.max_length_per_example
    if args.use_demonstrations:
        max_length = min(max_length_per_example * args.m, 2048)
    else:
        max_length = max_length_per_example

    logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (
        args.test_batch_size, max_length, max_length_per_example))

    metaicl_data = MetaICLData(logger, tokenizer, args.trunc_method, args.use_demonstrations, args.m,
                               max_length, max_length_per_example, is_opt_model=("opt" in args.gpt2))

    seed = args.seed # shorten name
    eval_data = test_data.values

    n_class = 2
    # test_task = eval_data[0]["task"]
    logger.info("-"*50)
    logger.info(f"Seed: {seed}, # Class: {n_class}")
    logger.info(f"[{args.split}]: {len(eval_data)}")

    logger.info(f"Eval {args.mode}")
    
    n_prompts = len(selected_samples)

    logger.info(f"# prompts: {n_prompts}")

    probs, perf = run(logger, args.rob_bound, args.num_class, dev_data.iloc[0]["options"], metaicl_data, metaicl_model, 
                                selected_samples, eval_data, 
                                seed, args.is_classification)
    
    print(perf)

   

def run(logger, rob_bound, num_class, options, metaicl_data, metaicl_model, train_data, eval_data, seed,
        is_classification):

    metaicl_data.tensorize_before(num_class, train_data, eval_data, options)
    metaicl_data.print_tensorized_example()

    probs = metaicl_model.do_inference(rob_bound, metaicl_data, args.test_batch_size, verbose=False)

    predictions, probs = metaicl_model.do_predict(metaicl_data, probs=probs)
    groundtruths = [dp[2] for dp in eval_data]
    perf = metaicl_data.evaluate(predictions, groundtruths, args.is_classification)
    logger.info("Accuracy=%s" % perf)

    return probs, perf



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_demonstrations", default= True, action="store_true")
    parser.add_argument('--is_classification', default = True, action='store_false')
    parser.add_argument("--trunc_method", type=str, default='right', choices=['right', 'left', 'middle'])
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--max_length_per_example", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--l", type=float, default=1.0)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default="final_results")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--gpt2", type=str, default="gpt-j")
    parser.add_argument("--mode", type=str, default="Random")
    parser.add_argument("--log_file", default=None, type=str)
    parser.add_argument("--num_class", default=2, type=int)
    parser.add_argument("--rob_bound", default=0.1, type=float)
    parser.add_argument("--train_path", default="", type=str)
    parser.add_argument("--test_path", default="", type=str)
    parser.add_argument("--dev_path", default="", type=str)
    parser.add_argument("--gene_value_path", default="", type=str)
    parser.add_argument("--rob_value_path", default="", type=str)


    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(logger, args)
