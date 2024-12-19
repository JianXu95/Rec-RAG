import os
import argparse
import json
from tqdm import trange, tqdm
import torch
# from fastchat.model import load_model, get_conversation_template, add_model_args
import numpy as np
import pandas as pd
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

cur_embed = None
embeds = []

def hook(module, input, output):
    global cur_embed, embeds
    input = input[0].cpu().detach().numpy()
    cur_embed = input


def simple_iter(args):
    if args.dataset == "ml-1m":
        movie_dict = json.load(open(os.path.join(args.data_dir, "movie_detail.json"), "r"))
        for i in trange(1, 3953):
            key = str(i)
            if key not in movie_dict.keys():
                title, genre = "", ""
            else:
                title, genre = movie_dict[key]
            text = \
                f"Here is a movie. Its title is {title}. The movie's genre is {genre}." 
            yield text

    elif args.dataset == "ml-25m":
        df_movie = pd.read_parquet(os.path.join(args.data_dir, 'ml_25m_movie_detail.parquet.gz'))

        for i in trange(len(df_movie)):
            row = df_movie.loc[i]
            text = \
                f"Here is a movie. Its title is {row['Movie title']}. The movie's genre is {row['Movie genre']}." 
            yield text
    
    elif args.dataset == "BookCrossing":
        id2book = json.load(open(os.path.join(args.data_dir, "id2book.json"), "r"))
        for i in trange(len(id2book)):
            isbn, title, author, year, publisher = id2book[str(i)]
            text = \
                f"Here is a book. Its title is {title}. ISBN of the book is {isbn}. The author of the book is {author}. "\
                f"The publication year of the book is {year}. Its publisher is {publisher}."
            yield text

    elif args.dataset == "amazon-movies":
        movie_dict = json.load(open(os.path.join(args.data_dir, "idx2movie.json"), "r"))
        for i in trange(1):
            key = str(i)
            if key not in movie_dict.keys():
                title, genre = "", ""
            else:
                movie_id, title, genre = movie_dict[key]
            text = \
                f"Here is a movie. Its title is {title}. The movie's genre is {genre}." 
            yield text
        
    else:
        assert False, "Unsupported dataset"


@torch.inference_mode()
def main(args):
    # Load model.
    # model, tokenizer = load_model(
    #     args.model_path,
    #     args.device,
    #     args.num_gpus,
    #     args.max_gpu_memory,
    #     args.load_8bit,
    #     args.cpu_offloading,
    #     # revision=args.revision,
    #     debug=args.debug,
    # )

    # Load model.
    transformers.set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, 
        add_eos_token=True, 
        padding_side="left",
        )

    # Ensure pad_token_id and attention_mask are set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad_token_id to eos_token_id if needed

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        # torch_dtype=torch.bfloat16, 
        # load_in_8bit=True,
        device_map="auto"
        )

    model.lm_head.register_forward_hook(hook)

    model.lm_head.register_forward_hook(hook)

    print("Model loaded.")

    os.makedirs(args.embed_dir, exist_ok=True)
    fp = os.path.join(args.embed_dir, '_'.join([args.dataset, args.pooling])+"_plain0.npy")


    global cur_embed, embeds

    # Start inference.
    # for txt in simple_iter(args):
    #     conv = get_conversation_template(args.model_path)
    #     conv.append_message(conv.roles[0], txt)
    #     conv.append_message(conv.roles[1], None)
    #     prompt = conv.get_prompt()

    #     cur_embed = None
    #     input_ids = tokenizer([prompt]).input_ids


    #     output_ids = model.generate(
    #         torch.as_tensor(input_ids).to(args.device),
    #         do_sample=True,
    #         temperature=args.temperature,
    #         repetition_penalty=args.repetition_penalty,
    #         max_new_tokens=args.max_new_tokens,
    #     )
        
        
    #     if args.pooling == "last":
    #         cur_embed = cur_embed[0, len(input_ids[0])-1]
    #     elif args.pooling == "average":
    #         cur_embed = cur_embed[0, :len(input_ids[0])].mean(axis=0)

    #     embeds.append(cur_embed)


    # np.save(fp, np.stack(embeds))

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
    # add_model_args(parser)
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
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2

    args.data_dir = f"../{args.data_dir}/{args.dataset}/proc_data"
    assert args.pooling in ["average", "last"], "Pooling type error"
    main(args)