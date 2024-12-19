import os
import argparse
import json
from tqdm import trange, tqdm
import torch
import numpy as np
import pandas as pd
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

cur_embed = None
embeds = []

def hook(module, input, output):
    global cur_embed, embeds
    input = input[0].cpu().detach().numpy()
    cur_embed = input


def simple_iter(args):

    start_index = args.start_index
    end_index = args.end_index
    assert start_index < end_index

    if args.dataset == "ml-1m":
        movie_dict = json.load(open(os.path.join(args.data_dir, "ml-1m_text.json"), "r"))
        for i in trange(1, 3953):
            key = str(i)
            if key not in movie_dict.keys():
                continue
            else:
                text = movie_dict[key]
            yield text

    elif args.dataset == "ml-25m":
        df_movie = pd.read_parquet(os.path.join(args.data_dir, 'ml_25m_movie_detail.parquet.gz'))

        for i in trange(len(df_movie)):
            row = df_movie.loc[i]
            text = \
                f"Here is a movie. Its title is {row['Movie title']}. The movie's genre is {row['Movie genre']}." 
            yield text
    
    elif args.dataset == "BookCrossing":
        book_dict = json.load(open(os.path.join(args.data_dir, "BookCrossing_text.json"), "r"))
        for i in trange(len(book_dict)):
            key = str(i)
            if key not in book_dict.keys():
                continue
            else:
                text = book_dict[key]
            yield text

    elif args.dataset == "amazon-movies":
        movie_dict = json.load(open(os.path.join(args.data_dir, "amazon-movies_text.json"), "r"))
        for i in trange(len(movie_dict)):
        # for i in range(start_index, end_index):
            key = str(i)
            if key not in movie_dict.keys():
                continue
            else:
                text = movie_dict[key]
            yield text
        
    else:
        assert False, "Unsupported dataset"


@torch.inference_mode()
def main(args):
    # Load model.
    transformers.set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, 
        add_eos_token=True, 
        padding_side="left",
        )

    # Ensure pad_token_id and attention_mask are set
    if tokenizer.pad_token_id is None:
        # tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad_token_id to eos_token_id if needed
        tokenizer.pad_token_id = 0

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        # torch_dtype=torch.bfloat16, 
        # load_in_8bit=True,
        device_map="auto"
        )

    model.lm_head.register_forward_hook(hook)

    print("Model loaded.")

    os.makedirs(args.embed_dir, exist_ok=True)
    fp = os.path.join(args.embed_dir, '_'.join([args.dataset, args.pooling])+".npy")
    # fp = os.path.join(args.embed_dir, '_'.join([args.dataset, args.pooling])+f"_{args.start_index}_{args.end_index}.npy")

    global cur_embed, embeds

    # Start inference.
    for txt in simple_iter(args):
        inputs = tokenizer(txt, return_tensors="pt").to('cuda')
        input_ids = inputs.input_ids

        cur_embed = None

        output_ids = model.generate(**inputs, do_sample=True, max_new_tokens=256)
        
        if args.pooling == "last":
            cur_embed = cur_embed[0, len(input_ids[0])-1]
        elif args.pooling == "average":
            cur_embed = cur_embed[0, :len(input_ids[0])].mean(axis=0)

        embeds.append(cur_embed)

    np.save(fp, np.stack(embeds))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pooling", type=str, default="average", help="average/last")
    parser.add_argument("--embed_dir", type=str, default="../embeddings")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="ml-1m", help="ml-1m/BookCrossing/amazon-movies")
    parser.add_argument("--model_path", type=str, default="/mnt/cache/sichunluo2/Llama-3.1-8B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--start_index', type=int, default=1, help="")
    parser.add_argument('--end_index', type=int, default=3953, help="")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2

    args.data_dir = f"../{args.data_dir}/{args.dataset}/proc_data"
    assert args.pooling in ["average", "last"], "Pooling type error"
    main(args)