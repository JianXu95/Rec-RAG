import pandas as pd
import os
import json
from tqdm import trange, tqdm
import numpy as np
import random

input_dict = {
    "User ID": None,
    "Movie ID": None,
    "history ID": None,
    "history rating": None,
    "sim_user_history": None,
}


def get_template(input_dict, temp_type="simple", sim_user=False):
    """
    The main difference of the prompts lies in the user behavhior sequence.
    simple: w/o retrieval
    sequential: w/ retrieval, the items keep their order in the original history sequence
    high: w/ retrieval, the items is listed with descending order of similarity to target item 
    """


    template = \
{
        "simple": 
f"The user watched the following movies in the past, and rated them:\n"
f"{list(map(lambda x: f'{x[1]}', enumerate(input_dict['history ID'])))}\n"
f"Note that more stars the user rated the movie, the user liked the movie more.\n"
f"And we think the user will like a new movie if the rating could be higher than 3 stars\n"
f"Based on the movies the user has watched, deduce if the user will like the movie ***{input_dict['Movie ID']}***.\n"
f"You should ONLY tell me yes or no.",


        "low": 
f"The user watched the following movies in the past, and rated them:\n"
f"{list(map(lambda x: f'{x[1]}', enumerate(input_dict['history ID'][::-1])))}\n"
f"Note that more stars the user rated the movie, the user liked the movie more.\n"
f"And we think the user will like a new movie if the rating could be higher than 3 stars\n"
f"Based on the movies the user has watched, deduce if the user will like the movie ***{input_dict['Movie ID']}***.\n"
f"You should ONLY tell me yes or no.",

        "high": 
f"The user watched the following movies in the past, and rated them:\n"
f"{list(map(lambda x: f'{x[1]}', enumerate(input_dict['history ID'][::])))}\n"
f"Note that more stars the user rated the movie, the user liked the movie more.\n"
f"And we think the user will like a new movie if the rating could be higher than 3 stars\n"
f"Based on the movies the user has watched, deduce if the user will like the movie ***{input_dict['Movie ID']}***.\n"
f"You should ONLY tell me yes or no.",

        "sequential": 
f"The user watched the following movies in the past, and rated them:\n"
f"{list(map(lambda x: f'{x[1]}', enumerate(input_dict['history ID'][::])))}\n"
f"Note that more stars the user rated the movie, the user liked the movie more.\n"
f"And we think the user will like a new movie if the rating could be higher than 3 stars\n"
f"Based on the movies the user has watched, deduce if the user will like the movie ***{input_dict['Movie ID']}***.\n"
f"You should ONLY tell me yes or no.",

        "rerank": 
f"The user watched the following movies in the past, and rated them:\n"
f"{list(map(lambda x: f'{x[1]}', enumerate(input_dict['history ID'][::])))}\n"
f"Note that more stars the user rated the movie, the user liked the movie more.\n"
f"And we think the user will like a new movie if the rating could be higher than 3 stars\n"
f"Based on the movies the user has watched, deduce if the user will like the movie ***{input_dict['Movie ID']}***.\n"
f"You should ONLY tell me yes or no.",

}        

    assert temp_type in template.keys(), "Template type error."
    # if sim_user and input_dict['sim_user_history'] is not None:

    #     template["sim_user"] =\
    #     f"The user is a {input_dict['Gender']}. "
    #     f"{objective} job is {input_dict['Job']}. {objective} age is {input_dict['Age']}.\n"
    #     f"{nominative.capitalize()} watched the following most relevant movies in the past, and rated them:\n"
    #     f"{list(map(lambda x: f'{x[1]}', enumerate(input_dict['history ID'][::])))}\n"
    #     f"As a reference, another similar user also watched the following relevant movies and rated them:\n"
    #     f"{list(map(lambda x: f'{x[1]}', enumerate(input_dict['sim_user_history'][::])))}\n"
    #     f"Based on the movies {nominative} has watched, deduce if {nominative} will like the movie ***{input_dict['Movie ID']}***.\n"
    #     f"Note that more stars the user rated the movie, the more the user liked the movie.\n"
    #     f"You should ONLY tell me yes or no.",

    #     return template["sim_user"]
    
    return template[temp_type]


def amazon_zero_shot_get_prompt(
    K=15, 
    temp_type="simple", 
    data_dir="../data/amazon-movies/proc_data", 
    istrain="test", 
    title_dir="id_to_title.json"
):
    global input_dict, template
    id_to_title = json.load(open(os.path.join(data_dir, title_dir), "r"))
    fp = f"{istrain}.parquet.gz"
    df = pd.read_parquet(os.path.join(data_dir, fp))

    # fill the template
    for index in trange((len(df))):
        cur_temp = row_to_prompt(index, df, K, id_to_title, temp_type)
        yield cur_temp



def amazon_zero_shot_ret_get_prompt(
    K=15,
    temp_type="simple", # sequential, high, rerank
    data_dir="../data/amazon-movies/proc_data", 
    istrain="test", 
    title_dir="id_to_title.json",
    emb_type="text",
    indice_dir="../embeddings",
    sim_user=False,
):
    global input_dict, template
    id_to_title = json.load(open(os.path.join(data_dir, title_dir), "r"))
    id_to_idx = json.load(open(os.path.join(data_dir, "id2idx.json"), "r"))
    id_to_movie = json.load(open(os.path.join(data_dir, "idx2movie.json"), "r"))
    fp = f"{istrain}.parquet.gz"
    df = pd.read_parquet(os.path.join(data_dir, fp))
    indice_path = os.path.join(indice_dir, '_'.join(["amazon-movies", emb_type, "indice"])+".npy")
    sorted_indice = np.load(indice_path)

    user_list = df["User ID"].values
    user_indice_path = os.path.join(indice_dir, '_'.join(["amazon-movies", "user", "indice"])+".npy")
    user_indice = np.load(user_indice_path)

    # fill the template
    for row_number in tqdm(list(df.index)):
        row = df.loc[row_number].to_dict()

        for key in input_dict:
            if key in row.keys(): # , "Key name error."
                input_dict[key] = row[key]


        cur_id = id_to_idx[input_dict["Movie ID"]]
        cur_indice = sorted_indice[cur_id,:]
        orig_hist_len = len(input_dict["history ID"])
        cnt = 0
        hist_rating_dict = {hist: rating  for hist, rating in zip(input_dict["history ID"], input_dict["history rating"])}
        if temp_type in ["sequential", "rerank"]:
            hist_seq_dict = {hist: i for i, hist in enumerate(input_dict["history ID"])}
            seq_hist_dict = {i: hist for i, hist in enumerate(input_dict["history ID"])}
            
        input_dict["history ID"], input_dict["history rating"] = [], []
        item_sim_thres = 0
        for idx in cur_indice:
            id = str(id_to_movie[str(idx)][0])
            if id in hist_rating_dict:
                cnt += 1
                input_dict["history ID"].append(id)
                input_dict["history rating"].append(hist_rating_dict[id])
                if cnt == 1:
                    item_sim_thres = np.argmax(cur_indice == idx)
                if cnt == K:
                    break

        assert len(input_dict["history ID"]) > 0, "error, no history"

        if temp_type == "sequential":
            zipped_list = sorted(zip(input_dict["history ID"], input_dict["history rating"]), key=lambda x: hist_seq_dict[x[0]])
            input_dict["history ID"], input_dict["history rating"] = map(list, zip(*zipped_list))

        if temp_type == "rerank" and orig_hist_len > K:
            K1 = int(2*K//3)
            K2 = K - K1
            zipped_list = sorted(zip(input_dict["history ID"], input_dict["history rating"]), key=lambda x: -hist_seq_dict[x[0]])
            history_id, history_rating = map(list, zip(*zipped_list))
            hist_length = len(history_id)
            cnt = 0
            rerank_hist = []
            rerank_rate = []
            while len(rerank_hist) < K1 and cnt < hist_length:
                rerank_hist.append(history_id[cnt])
                rerank_rate.append(history_rating[cnt])
                cnt += 1
            input_dict["history ID"] = rerank_hist
            input_dict["history rating"] = rerank_rate
            # input_dict["history ID"] = input_dict["history ID"][:K1] #+ history_id[:K2]
            # input_dict["history rating"] = input_dict["history rating"][:K1] #+ history_rating[:K2]
            hist_length = len(hist_rating_dict)
            cnt = 0
            recent_hist = []
            recent_rate = []
            while len(recent_hist) < K2 and cnt < hist_length:
                hist_id = seq_hist_dict[hist_length-1-cnt]
                if hist_id not in input_dict["history ID"]:
                    recent_hist.append(hist_id)
                    recent_rate.append(hist_rating_dict[hist_id])
                cnt += 1
            recent_hist.reverse()
            recent_rate.reverse()
            input_dict["history ID"] = input_dict["history ID"] + recent_hist
            input_dict["history rating"] = input_dict["history rating"] + recent_rate

        # if temp_type == "rerank":
        #     zipped_list = sorted(zip(input_dict["history ID"], input_dict["history rating"]), key=lambda x: hist_seq_dict[x[0]])
        #     history_id, history_rating = map(list, zip(*zipped_list))
        #     hist_length = len(history_id)
        #     cnt = 0
        #     rerank_hist = []
        #     rerank_rate = []
        #     while len(rerank_hist) < K and cnt < hist_length:
        #         if history_id[cnt] in input_dict["history ID"]:
        #             rerank_hist.append(history_id[cnt])
        #             rerank_rate.append(history_rating[cnt])
        #         cnt += 1
        #     input_dict["history ID"] = rerank_hist
        #     input_dict["history rating"] = rerank_rate

        if sim_user:
            input_dict["sim_user_history"] = None
            cur_user_id = int(input_dict["User ID"])
            sim_user_indice = user_indice[cur_user_id,:]
            sim_user_id = sim_user_indice[1]
            for id in sim_user_indice:
                if id != cur_user_id and id in user_list:
                    sim_user_id = id
                    break

            sim_user_df = df[df['User ID'] == sim_user_id].tail(1).iloc[0].to_dict()
            hist_rating_sim = {hist: rating  for hist, rating in zip(sim_user_df["history ID"], sim_user_df["history rating"])}
            sim_history, sim_rating = [], []
            cnt = 0
            for index in cur_indice:
                # index = str(index)
                sim_order = np.argmax(cur_indice == index)
                if index in hist_rating_sim and sim_order < item_sim_thres:
                    cnt += 1
                    sim_history.append(index)
                    sim_rating.append(hist_rating_sim[index])
                    if cnt == 3:
                        break
            # sim_history = list(map(lambda index: id_to_title[str(index)], sim_history))
            # for i, (name, star) in enumerate(zip(sim_history, sim_rating)):
            #     suffix = " stars)" if star > 1 else " star)"
            #     # here changed
            #     sim_history[i] = f"{name} ({star}" + suffix
            # input_dict["sim_user_history"] = sim_history
            sim_length = len(sim_history)
            input_dict["history ID"] = sim_history + input_dict["history ID"][sim_length:]
            input_dict["history rating"] = sim_rating + input_dict["history rating"][sim_length:]

            # if row_number < 10:
            #     print(sim_history)

        movie_id = str(input_dict["Movie ID"])
        input_dict["Movie ID"] = id_to_title[movie_id]
        input_dict["history ID"] = list(map(lambda index: id_to_title[str(index)], input_dict["history ID"]))

        for i, (name, star) in enumerate(zip(input_dict["history ID"], input_dict["history rating"])):
            suffix = " stars)" if star > 1 else " star)"
            # here changed
            input_dict["history ID"][i] = f"{name} ({star}" + suffix

        yield get_template(input_dict, temp_type, sim_user)


def amazon_hybrid_ret_get_prompt(
    K=15,
    temp_type="hybrid", # sequential, high, rerank
    data_dir="../data/amazon-movies/proc_data", 
    istrain="test", 
    title_dir="id_to_title.json",
    emb_type="hybrid",
    indice_dir="../embeddings",
):
    assert emb_type == "hybrid"
    global input_dict, template
    id_to_title = json.load(open(os.path.join(data_dir, title_dir), "r"))
    id_to_idx = json.load(open(os.path.join(data_dir, "id2idx.json"), "r"))
    id_to_movie = json.load(open(os.path.join(data_dir, "idx2movie.json"), "r"))
    fp = f"{istrain}.parquet.gz"
    df = pd.read_parquet(os.path.join(data_dir, fp))
    indice_dir_1 = os.path.join(indice_dir, '_'.join(["amazon-movies", "text", "indice"])+".npy")
    indice_dir_2 = os.path.join(indice_dir, '_'.join(["amazon-movies", "colla", "indice"])+".npy")
    sorted_indice_1 = np.load(indice_dir_1)
    sorted_indice_2 = np.load(indice_dir_2)

    # fill the template
    for row_number in tqdm(list(df.index)):
        row = df.loc[row_number].to_dict()

        for key in input_dict:
            if key in row.keys(): #"Key name error."
                input_dict[key] = row[key]


        cur_id = id_to_idx[input_dict["Movie ID"]]
        K1 = (K+1)//2
        K2 = K - K1
        hist_rating_dict = {hist: rating  for hist, rating in zip(input_dict["history ID"], input_dict["history rating"])}
            
        input_dict["history ID"], input_dict["history rating"] = [], []
        cur_indice = sorted_indice_1[cur_id,:]
        cnt = 0
        for index in cur_indice:
            id = str(id_to_movie[str(index)][0])
            if index in hist_rating_dict:
                cnt += 1
                input_dict["history ID"].append(id)
                input_dict["history rating"].append(hist_rating_dict[id])
                if cnt == K1:
                    break
        cur_indice = sorted_indice_2[cur_id,:]
        cnt = 0
        for index in cur_indice:
            id = str(id_to_movie[str(index)][0])
            if index in hist_rating_dict and index not in input_dict["history ID"]:
                cnt += 1
                input_dict["history ID"].append(id)
                input_dict["history rating"].append(hist_rating_dict[id])
                if cnt == K2:
                    break

        movie_id = str(input_dict["Movie ID"])
        input_dict["Movie ID"] = id_to_title[movie_id]
        input_dict["history ID"] = list(map(lambda index: id_to_title[str(index)], input_dict["history ID"]))

        for i, (name, star) in enumerate(zip(input_dict["history ID"], input_dict["history rating"])):
            suffix = " stars)" if star > 1 else " star)"
            # here changed
            input_dict["history ID"][i] = f"{name} ({star}" + suffix

        yield get_template(input_dict, temp_type)


def row_to_prompt(index, df, K, id_to_title, temp_type="simple"):
    global input_dict, template
    row = df.loc[index].to_dict()

    for key in input_dict:
        if key in row.keys(): #, "Key name error."
            input_dict[key] = row[key]

    # convert history ID from id to name
    movie_id = str(input_dict["Movie ID"])
    input_dict["Movie ID"] = id_to_title[movie_id]
    input_dict["history ID"] = list(map(lambda x: id_to_title[str(x)], input_dict["history ID"]))

    input_dict["history ID"] = input_dict["history ID"][-K:]
    input_dict["history rating"] = input_dict["history rating"][-K:]
    for i, (name, star) in enumerate(zip(input_dict["history ID"], input_dict["history rating"])):
        suffix = " stars)" if star > 1 else " star)"
        input_dict["history ID"][i] = f"{name} ({star}" + suffix

    return get_template(input_dict, temp_type)