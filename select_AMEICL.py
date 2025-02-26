import numpy as np
import argparse
import pickle as pkl
import itertools
from glob import glob
import os
from utils.selection import *
from config.config import OUT_SELECT
import pandas as pd
import tqdm
from transformers import GPT2Tokenizer, AutoTokenizer
from metaicl.model import MetaICLModel
from metaicl.data import MetaICLData
import logging
from utils.data import load_data, random_subset_of_comb
from scipy.stats import zscore
from sklearn.linear_model import ElasticNetCV



parser = argparse.ArgumentParser()
parser.add_argument("--use_demonstrations", default=True, action="store_true")
parser.add_argument('--is_classification', default=True, action='store_false')
parser.add_argument("--trunc_method", type=str, default='right', choices=['right', 'left', 'middle'])
parser.add_argument("--source_task", type=str, default=None, help="prompts from a different task; OOD experiments")
parser.add_argument("--m", type=int, default=8)
parser.add_argument("--max_length_per_example", type=int, default=128)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--test_batch_size", type=int, default=16)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--gpt2", type=str, default="gpt-j")
parser.add_argument("--mode", type=str, default="Random")
parser.add_argument("--log_file", default=None, type=str)
parser.add_argument("--s_n", default=100, type=int)
parser.add_argument("--num_class", default=2, type=int)
parser.add_argument("--train_path", default="", type=str)
parser.add_argument("--test_path", default="", type=str)
parser.add_argument("--dev_path", default="", type=str)
parser.add_argument("--model_path", default="", type=str)
args = parser.parse_args()

train_data = pd.read_json(args.train_path, lines=True)
test_data = pd.read_json(args.test_path, lines=True)
dev_data = pd.read_json(args.dev_path, lines=True)
num_points = len(train_data)

proportions = [0.2, 0.4, 0.6, 0.8]
num_subsets = args.s_n 

performance = np.zeros((num_subsets,)) 


tokenizer = GPT2Tokenizer.from_pretrained(args.model_path) 
checkpoint = None
logger = logging.getLogger(__name__)
logger.info(args)
metaicl_model = MetaICLModel(logger)
metaicl_model.load(checkpoint, gpt2=args.gpt2)
metaicl_model.to_device()
metaicl_model.eval()

max_length_per_example = args.max_length_per_example
if args.use_demonstrations:
    max_length = min(max_length_per_example * args.m, 2048)
else:
    max_length = max_length_per_example

logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (
        args.test_batch_size, max_length, max_length_per_example))


def run(logger, num_class, options, metaicl_data, metaicl_model, train_data, eval_data, seed,
        is_classification):

    metaicl_data.tensorize_before(num_class, train_data, eval_data, options)
    metaicl_data.print_tensorized_example()

    probs = metaicl_model.do_inference(metaicl_data, args.test_batch_size, verbose=False)

    predictions, probs = metaicl_model.do_predict(metaicl_data, probs=probs)

    groundtruths = [dp[2] for dp in eval_data]
    perf = metaicl_data.evaluate(predictions, groundtruths, args.is_classification)
    logger.info("Accuracy=%s" % perf)

    return probs, perf

subsets_total, performance_total = [], []

for proportion in proportions:
    sample_dim = (num_subsets, num_points)
    subsets = np.random.binomial(1, proportion, size=sample_dim)
    for i in tqdm.tqdm(range(num_subsets)):
        subset = subsets[i].nonzero()[0]
        if not subset.any():
            continue
        selected_samples = train_data.values[subset,:]
        metaicl_data = MetaICLData(logger, tokenizer, args.trunc_method, args.use_demonstrations, args.m,
                               max_length, max_length_per_example, is_opt_model=False)

        seed = args.seed # shorten name
        eval_data = dev_data.values

        n_class = args.num_class 
        logger.info("-"*50)
        logger.info(f"Seed: {seed},  # Class: {n_class}")
        logger.info(f"[{args.split}]: {len(eval_data)}")

        logger.info(f"Eval {args.mode}")

        
        n_prompts = len(selected_samples)

        logger.info(f"# prompts: {n_prompts}")


        all_probs, all_perf = run(logger, args.num_class, dev_data.iloc[0]["options"], metaicl_data, metaicl_model, 
                                selected_samples, eval_data, 
                                seed, args.is_classification)

     
        performance[i] = all_perf 
    subsets_total.append(subsets)
    performance_total.append(performance)

subsets_t = np.concatenate(subsets_total)
performance_t = np.concatenate(performance_total)
print(performance_t)
print(subsets_t)



norm_subsets = zscore(subsets_t, axis=1)
norm_subsets[np.isnan(norm_subsets)] = 0
centered_perf = performance_t - np.mean(performance_t)

dv_ame = ElasticNetCV()
dv_ame.fit(X=norm_subsets, y=centered_perf) #

data_values = dv_ame.coef_

print(data_values)


file_path = "AME-ICL/"+str(args.s_n)+".npy"
np.save(file_path, data_values)
