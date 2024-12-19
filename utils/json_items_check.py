import json, os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--K", type=int, default=30)
parser.add_argument("--train_type", type=str, default="mixed", help="simple/mixed")
parser.add_argument("--temp_type", type=str, default="high", help="sequtial/high")
parser.add_argument("--emb_type", type=str, default="mix3", help="text/colla/mix/mix3")
parser.add_argument("--dataset", type=str, default="ml-1m")
parser.add_argument("--train_size", type=int, default=1024)
parser.add_argument("--val_size", type=int, default=1000)
parser.add_argument("--set", type=str, default="test", help="train/valid/test")
parser.add_argument("--save", type=int, default=1, help="train/valid/test")
args = parser.parse_args()
print(args)

data_path = f"../data/{args.dataset}/proc_data/data"

assert args.temp_type in ["sequential", "high", "rerank", "simple"]
assert args.set in ["train", "test"]


DATA_PATH = {
    "train": '/'.join([data_path, f"train/train_{args.K}_{args.temp_type}_{args.emb_type}.json"]), 
    "test": '/'.join([data_path, f"test/test_{args.K}_{args.temp_type}_{args.emb_type}.json"])
}

data = load_dataset("json", data_files=DATA_PATH)
# data["train"] = data["train"].select(range(args.train_size))
# data["test"] = data["test"].select(range(args.val_size))
train_data = data['train']
test_data = data['test']

eval_size = 1
cnt = 0
for data_point in tqdm(train_data):
    if cnt >= eval_size:
        break
    cnt += 1
    print(data_point)
