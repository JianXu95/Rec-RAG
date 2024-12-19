import json, os
import argparse
import pandas as pd
import random


parser = argparse.ArgumentParser()
parser.add_argument("--K", type=int, default=15)
parser.add_argument("--temp_type", type=str, default="high")
parser.add_argument("--emb_type", type=str, default="text")
parser.add_argument("--dataset", type=str, default="ml-1m")
parser.add_argument("--set", type=str, default="train", help="train/valid/test")
args = parser.parse_args()

DATA_DIR = f"../data/{args.dataset}/proc_data/data"


data_set = json.load(open('/'.join([DATA_DIR, f"{args.set}/{args.set}_{args.K}_{args.temp_type}_{args.emb_type}.json"])))
random.seed(42)
fewshot = 256 if args.dataset == "BookCrossing" else 8192
sample_index = random.sample(range(len(data_set)), fewshot)
# 256 512 1024 4096 8192 65536

for set in [args.set]:
    for K in [args.K]:
        for temp_type in [ "high", "sequential", "simple"]:
            print(f"==> {set}, {K}, {temp_type}")
            
            args.K = K
            args.set = set
            args.temp_type = temp_type
            print(args)

            # assert args.temp_type in ["high", "sequential"]

            fp = '/'.join([DATA_DIR, f"{args.set}/{args.set}_{args.K}_{args.temp_type}_{args.emb_type}.json"])
            data = json.load(open(fp, 'r'))
            indice = sample_index
            sampled = [data[i] for i in indice]
            json.dump(sampled, open('/'.join([DATA_DIR, f"{args.set}/{args.set}_{args.K}_{args.temp_type}_{args.emb_type}_sampled.json"]), "w"), indent=4)
            print("  Dumped.")
            
if args.set in ["train"]:
    set = args.set
    for K in [args.K]:
        a = json.load(open(f"{DATA_DIR}/{set}/{set}_{K}_high_{args.emb_type}_sampled.json"))
        print(len(a))
        b = json.load(open(f"{DATA_DIR}/{set}/{set}_{K}_sequential_{args.emb_type}_sampled.json"))
        print(len(b))
        c = json.load(open(f"{DATA_DIR}/{set}/{set}_{K}_simple_text_sampled.json"))
        print(len(c))
        t = []
        for m, n, l in zip(a, b, c):
            t.append(m)
            t.append(n)
            t.append(l)
        print(len(t))
        json.dump(t, open(f"{DATA_DIR}/{set}/{set}_{K}_mixed_{args.emb_type}_sampled.json", "w"), indent=4)

        t = []
        for m, n in zip(a, b):
            t.append(m)
            t.append(n)
        print(len(t))
        json.dump(t, open(f"{DATA_DIR}/{set}/{set}_{K}_rag_{args.emb_type}_sampled.json", "w"), indent=4)

