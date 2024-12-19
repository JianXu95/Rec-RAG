import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os, argparse
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

parser = argparse.ArgumentParser()
parser.add_argument("--set", type=str, default="test")
parser.add_argument("--embed_dir", type=str, default="../embeddings")
parser.add_argument("--embed_type", type=str, default="text", help="txt/text/colla/mix/mix3")
parser.add_argument("--pooling", type=str, default="average")
parser.add_argument("--data_dir", type=str, default="../data/BookCrossing/proc_data")
parser.add_argument("--dim", type=int, default=128)
args = parser.parse_args()


def item_sim(args):
    fp0 = f"../embeddings/BookCrossing_{args.pooling}_plain.npy" 
    fp1 = f"../embeddings/BookCrossing_{args.pooling}.npy" 
    fp2 = "../data/BookCrossing/saved_embed/lightgcn_item_emb.npy"
    fp3 = f"../embeddings/BookCrossing_{args.pooling}_ssl.npy" 
    colla_id2item = json.load(open("../data/BookCrossing/saved_embed/lightgcn_id2item.json", "r"))

    embed1 = np.load(fp1)
    n_item = embed1.shape[0]

    if args.embed_type == "txt":
        embed1 = np.load(fp0)
        pca = PCA(n_components=args.dim)
        embed = pca.fit_transform(embed1)

    if args.embed_type not in ["txt", "text"]:
        colla_emb = np.load(fp2)
        isbn2id = json.load(open("../data/BookCrossing/proc_data/isbn2id.json"))
        _, colla_dim = colla_emb.shape
        embed2 = np.zeros((n_item, colla_dim))
        for emb_id, item_id in colla_id2item.items():
            embed2[int(isbn2id[item_id])] = colla_emb[int(emb_id)]
        # embed = np.concatenate((embed1, embed2), axis=1)

    if args.embed_type == "colla":
        embed = embed2
    
    if args.embed_type in ["mix", "mix2"]:
        pca_1 = PCA(n_components=args.dim)
        embed_1 = pca_1.fit_transform(embed1)
        # pca_2 = PCA(n_components=args.dim)
        # embed_2 = pca_2.fit_transform(embed2)
        embed_2 = embed2
        row_norms_text = np.linalg.norm(embed_1, ord=2, axis=1)
        new_embed_1 = embed_1/np.max(row_norms_text)
        row_norms_colla = np.linalg.norm(embed_2, ord=2, axis=1)
        new_embed_2 = embed_2/np.max(row_norms_colla)
        embed = np.concatenate((new_embed_1, new_embed_2), axis=1)

    if args.embed_type == "mix3":
        pca_1 = PCA(n_components=args.dim)
        embed_1 = pca_1.fit_transform(embed1)
        # pca_2 = PCA(n_components=args.dim)
        # embed_2 = pca_2.fit_transform(embed2)
        embed_2 = embed2
        embed_3 = np.load(fp3)
        row_norms_text = np.linalg.norm(embed_1, ord=2, axis=1)
        new_embed_1 = embed_1/np.max(row_norms_text)
        row_norms_colla = np.linalg.norm(embed_2, ord=2, axis=1)
        new_embed_2 = embed_2/np.max(row_norms_colla)
        new_embed_1 = embed_1/np.max(row_norms_text)
        row_norms_colla_ssl = np.linalg.norm(embed_3, ord=2, axis=1)
        new_embed_3 = embed_3/np.max(row_norms_colla_ssl)
        embed = np.concatenate((new_embed_1, new_embed_2, new_embed_3), axis=1)

    elif args.embed_type == "text":
        pca = PCA(n_components=args.dim)
        embed = pca.fit_transform(embed1)
    
    print(embed.shape)

    # sim_matrix = cosine_similarity(embed)
    # print("Similarity matrix computed.")

    # sorted_indice = np.argsort(-sim_matrix, axis=1)
    # print("Sorted.")

    # fp_indice =os.path.join(args.embed_dir, '_'.join(["BookCrossing", args.embed_type, "indice"])+".npy")
    # np.save(fp_indice, sorted_indice)
    # print("Saved.")
    return embed


def user_sim(args):
    fp = "../data/BookCrossing/saved_embed/lightgcn_user_emb.npy"
    embed = np.load(fp)
    print(embed.shape)

    sim_matrix = cosine_similarity(embed)
    print("Similarity matrix computed.")

    sorted_indice = np.argsort(-sim_matrix, axis=1)
    print("Sorted.")

    fp_indice =os.path.join(args.embed_dir, '_'.join(["BookCrossing", "user", "indice"])+".npy")
    np.save(fp_indice, sorted_indice)
    print("Saved.")

# user_sim(args)

isbn2id = json.load(open(f"{args.data_dir}/isbn2id.json"))
id2book = json.load(open(f"{args.data_dir}/id2book.json"))
# embeddings = np.load(f"./embeddings/BookCrossing_{args.pooling}.npy")
embeddings = item_sim(args)
print("Embeddings loaded.")
print(embeddings.shape)

all_indice = []

df = pd.read_parquet(f"{args.data_dir}/{args.set}.parquet.gz")
df = df.reset_index(drop=True)

for idx, row in tqdm(df.iterrows()):
    tgt_id = isbn2id[row['ISBN']]
    hist_id = [isbn2id[isbn] for isbn in row['user_hist']]
    # hist_id = [id for id in row['user_hist']]
    
    tgt_embed, hist_embed = embeddings[tgt_id], embeddings[hist_id]
    
    seq_id_to_book_id = {i: book_id for i, book_id in enumerate(hist_id)}
    sim_matrix = np.sum(hist_embed * tgt_embed, axis=-1)
    if args.embed_type == "mix2":
        tgt_embed1, tgt_embed2 = tgt_embed[:args.dim], tgt_embed[args.dim:]
        hist_embed1, hist_embed2 = hist_embed[:, :args.dim], hist_embed[:, args.dim:]
        sim_matrix = 0.5*softmax(np.sum(hist_embed1 * tgt_embed1, axis=-1)) + 0.5*softmax(np.sum(hist_embed2 * tgt_embed2, axis=-1))
    indice = np.argsort(-sim_matrix)[:100].tolist()
    sorted_indice = list(map(lambda x: id2book[str(seq_id_to_book_id[x])][0], indice))
    all_indice.append(sorted_indice)

json.dump(all_indice, open(f'../embeddings/BookCroosing_{args.embed_type}_indice_{args.set}.json', 'w'), indent=4)