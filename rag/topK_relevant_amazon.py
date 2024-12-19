import os, argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from scipy.special import softmax
from tqdm import trange
import json
# import faiss


def item_sim(args):
    fp0 = f"../embeddings/amazon-movies_{args.pooling}_plain.npy" 
    fp1 = f"../embeddings/amazon-movies_{args.pooling}.npy" 
    fp2 = "../data/amazon-movies/saved_embed/lightgcn_item_emb.npy"
    # fp3 = "../data/amazon-movies/saved_embed/lightgcn_item_emb_ssl.npy"
    fp3 = f"../embeddings/amazon-movies_{args.pooling}_ssl.npy" 
    colla_id2item = json.load(open("../data/amazon-movies/saved_embed/lightgcn_id2item.json", "r"))

    embed1 = np.load(fp1)
    n_item = embed1.shape[0]

    if args.embed_type == "txt":
        embed = np.load(fp0)

    if args.embed_type not in ["txt", "text"]:
        colla_emb = np.load(fp2)
        id2idx = json.load(open("../data/amazon-movies/proc_data/id2idx.json"))
        _, colla_dim = colla_emb.shape
        embed2 = np.zeros((n_item, colla_dim))
        for emb_id, item_id in colla_id2item.items():
            embed2[int(id2idx[item_id])] = colla_emb[int(emb_id)]
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

    sim_matrix = cosine_similarity(embed)
    print("Similarity matrix computed.")

    if args.embed_type == "mix2":
        sim_matrix = 0.5*softmax(cosine_similarity(new_embed_1), axis=1) + 0.5*softmax(cosine_similarity(new_embed_2), axis=1)

    sorted_indice = np.argsort(-sim_matrix, axis=1)
    print("Sorted.")

    fp_indice =os.path.join(args.embed_dir, '_'.join(["amazon-movies", args.embed_type, "indice"])+".npy")
    np.save(fp_indice, sorted_indice)
    print("Saved.")
    


def item_sim_ssl(args):
    fp1 = f"../embeddings/amazon-movies_{args.pooling}_ssl.npy" 
    fp2 = "../data/amazon-movies/saved_embed/lightgcn_item_emb_ssl.npy"
    colla_id2item = json.load(open("../data/amazon-movies/saved_embed/lightgcn_id2item.json", "r"))

    embed1 = np.load(fp1)
    n_item = embed1.shape[0]

    if args.embed_type != "text":
        colla_emb = np.load(fp2)
        _, colla_dim = colla_emb.shape
        embed2 = np.zeros((n_item, colla_dim))
        for emb_id, item_id in colla_id2item.items():
            embed2[int(item_id)-1] = colla_emb[int(emb_id)]
        print(f"dim of colla emb: {colla_dim}")

        # embed = np.concatenate((embed1, embed2), axis=1)

    if args.embed_type == "colla":
        embed = embed2
    
    if args.embed_type == "mix":
        pca = PCA(n_components=128)
        embed_1 = pca.fit_transform(embed1)
        # embed_1 = embed1
        embed_2 = embed2
        row_norms_text = np.linalg.norm(embed_1, ord=2, axis=1)
        new_embed_1 = embed_1/np.max(row_norms_text)
        row_norms_colla = np.linalg.norm(embed_2, ord=2, axis=1)
        new_embed_2 = embed_2/np.max(row_norms_colla)
        embed = np.concatenate((new_embed_1, new_embed_2), axis=1)

    elif args.embed_type == "text":
        embed = embed1
    
    print(embed.shape)

    sim_matrix = cosine_similarity(embed)
    print("Similarity matrix computed.")

    sorted_indice = np.argsort(-sim_matrix, axis=1)
    print("Sorted.")

    fp_indice =os.path.join(args.embed_dir, '_'.join(["amazon-movies", args.embed_type, "indice"])+"_ssl.npy")
    np.save(fp_indice, sorted_indice)
    print("Saved.")
    
def user_sim(args):
    fp = "../data/amazon-movies/saved_embed/lightgcn_user_emb.npy"
    embed = np.load(fp)
    print(embed.shape)

    sim_matrix = cosine_similarity(embed)
    print("Similarity matrix computed.")

    sorted_indice = np.argsort(-sim_matrix, axis=1)
    print("Sorted.")

    fp_indice =os.path.join(args.embed_dir, '_'.join(["amazon-movies", "user", "indice"])+".npy")
    np.save(fp_indice, sorted_indice)
    print("Saved.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dir", type=str, default="../embeddings")
    parser.add_argument("--embed_type", type=str, default="text", help="txt/text/colla/mix/mix3")
    parser.add_argument("--pooling", type=str, default="average", help="average/last")
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()
    item_sim(args)
    # user_sim(args)
    # item_sim_ssl(args)

