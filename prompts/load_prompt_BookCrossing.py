import pandas as pd
import os
import json
from tqdm import trange, tqdm
import numpy as np

input_dict = {
    "User ID": None,
    "Location": None,
    "Age": None,
    "Book title": None,
    "user_hist": None,
    "hist_rating": None,
}



def get_template(input_dict, temp_type="simple"):
    """
    The main difference of the prompts lies in the user behavhior sequence.
    simple: w/o retrieval
    sequential: w/ retrieval, the items keep their order in the original history sequence
    high: w/ retrieval, the items is listed with descending order of similarity to target item 
    """

    template = \
{
        "simple": 
f"The user's location is {input_dict['Location']}. The user's age is {input_dict['Age']}.\n"
f"The user read the following books in the past and rated them:\n"
f"{list(map(lambda x: f'{x[1]}', enumerate(input_dict['user_hist'])))}\n"
f"Based on the books the user has read, deduce if the user will like the book ***{input_dict['Book title']}***.\n"
f"Note that more stars the user rated the book, the user liked the book more.\n"
f"And we think the user will like a new book if the rating could be higher than 4 stars\n"
f"You should ONLY tell me yes or no.",

        "sequential": 
f"The user's location is {input_dict['Location']}. The user's age is {input_dict['Age']}.\n"
f"The user read the following books in the past and rated them:\n"
f"{list(map(lambda x: f'{x[1]}', enumerate(input_dict['user_hist'][::])))}\n"
f"Based on the books the user has read, deduce if the user will like the book ***{input_dict['Book title']}***.\n"
f"Note that more stars the user rated the book, the user liked the book more.\n"
f"And we think the user will like a new book if the rating could be higher than 4 stars\n"
f"You should ONLY tell me yes or no.",

        "low": 
f"The user's location is {input_dict['Location']}. The user's age is {input_dict['Age']}.\n"
f"The user read the following books in the past and rated them:\n"
f"{list(map(lambda x: f'{x[0]}. {x[1]}', enumerate(input_dict['user_hist'][::-1])))}\n"
f"Based on the books the user has read, deduce if the user will like the book ***{input_dict['Book title']}***.\n"
f"Note that more stars the user rated the book, the user liked the book more.\n"
f"You should ONLY tell me yes or no.",

        "high": 
f"The user's location is {input_dict['Location']}. The user's age is {input_dict['Age']}.\n"
f"The user read the following books in the past and rated them:\n"
f"{list(map(lambda x: f'{x[1]}', enumerate(input_dict['user_hist'][::])))}\n"
f"Based on the books the user has read, deduce if the user will like the book ***{input_dict['Book title']}***.\n"
f"Note that more stars the user rated the book, the user liked the book more.\n"
f"And we think the user will like a new book if the rating could be higher than 4 stars\n"
f"You should ONLY tell me yes or no.",

        "rerank": 
f"The user's location is {input_dict['Location']}. The user's age is {input_dict['Age']}.\n"
f"The user read the following books in the past and rated them:\n"
f"{list(map(lambda x: f'{x[1]}', enumerate(input_dict['user_hist'][::])))}\n"
f"Based on the books the user has read, deduce if the user will like the book ***{input_dict['Book title']}***.\n"
f"Note that more stars the user rated the book, the user liked the book more.\n"
f"And we think the user will like a new book if the rating could be higher than 4 stars\n"
f"You should ONLY tell me yes or no.",
}

    assert temp_type in template.keys(), "Template type error."
    return template[temp_type]


def book_zero_shot_get_prompt(
    K=15, 
    temp_type="simple", 
    data_dir="./data/BookCrossing/proc_data", 
    istrain="test",
):
    global input_dict, template
    fp = f"{istrain}.parquet.gz"
    df = pd.read_parquet(os.path.join(data_dir, fp))

    id2book = json.load(open(os.path.join(data_dir, "id2book.json"), "r"))
    isbn2id = json.load(open(os.path.join(data_dir, "isbn2id.json"), "r"))
    isbn2title = {isbn: id2book[str(isbn2id[isbn])][1] for isbn in isbn2id.keys()}


    # fill the template
    for index in tqdm(list(df.index)):
        cur_temp = row_to_prompt(index, df, K, isbn2title, temp_type)
        yield cur_temp



def book_zero_shot_ret_get_prompt(
    K=15,
    temp_type="simple", 
    data_dir="../data/BookCrossing/proc_data", 
    istrain="test", 
    emb_type="text",
):
    global input_dict, template
    fp = f"{istrain}.parquet.gz"
    df = pd.read_parquet(os.path.join(data_dir, fp)).reset_index(drop=True)
    indice_dir = f"../embeddings/BookCroosing_{emb_type}_indice_{istrain}.json"
    sorted_indice = json.load(open(indice_dir))

    id2book = json.load(open(os.path.join(data_dir, "id2book.json"), "r"))
    isbn2id = json.load(open(os.path.join(data_dir, "isbn2id.json"), "r"))
    isbn2title = {isbn: id2book[str(isbn2id[isbn])][1] for isbn in isbn2id.keys()}
    # id2title = {id: id2book[id][1] for id in id2book.keys()}


    # fill the template
    for row_number in tqdm(list(df.index)):
        row = df.loc[row_number].to_dict()
        # print(row)

        for key in input_dict:
            if key in row.keys(): #, "Key name error."
                input_dict[key] = row[key]

        cur_indice = sorted_indice[row_number]
        orig_hist_len = len(input_dict["user_hist"])
        hist_rating_dict = {hist: rating  for hist, rating in zip(input_dict["user_hist"], input_dict["hist_rating"])}
        if temp_type in ["sequential", "rerank"]:
            hist_seq_dict = {hist: i for i, hist in enumerate(input_dict["user_hist"])}
            seq_hist_dict = {i: hist for i, hist in enumerate(input_dict["user_hist"])}
            
        input_dict["user_hist"], input_dict["hist_rating"] = [], []
        
        for i in range(min(K, len(cur_indice))):
           input_dict['user_hist'].append(cur_indice[i])
           input_dict['hist_rating'].append(hist_rating_dict[cur_indice[i]])

        if temp_type == "sequential":
            zipped_list = sorted(zip(input_dict["user_hist"], input_dict["hist_rating"]), key=lambda x: hist_seq_dict[x[0]])
            input_dict["user_hist"], input_dict["hist_rating"] = map(list, zip(*zipped_list))
        # input_dict["user_hist"] = list(map(lambda isbn: isbn2title[isbn], input_dict["user_hist"]))
        # input_dict["user_hist"] = list(map(lambda id: id2title[id], input_dict["user_hist"]))

        if temp_type == "rerank" and orig_hist_len > K:
            K1 = int(2*K//3)
            K2 = K - K1
            # input_dict["user_hist"] = input_dict["user_hist"][:K1] #+ history_id[:K2]
            # input_dict["hist_rating"] = input_dict["hist_rating"][:K1] #+ history_rating[:K2]
            # hist_length = len(hist_rating_dict)
            # cnt = 0
            # recent_hist = []
            # recent_rate = []
            # while len(recent_hist) < K2 and cnt < hist_length:
            #     hist_id = seq_hist_dict[cnt]
            #     if hist_id not in input_dict["user_hist"]:
            #         recent_hist.append(hist_id)
            #         recent_rate.append(hist_rating_dict[hist_id])
            #     cnt += 1
            # input_dict["user_hist"] = input_dict["user_hist"] + recent_hist
            # input_dict["hist_rating"] = input_dict["hist_rating"] + recent_rate

            # zipped_list = sorted(zip(input_dict["user_hist"], input_dict["hist_rating"]), key=lambda x: hist_seq_dict[x[0]])
            # history_id, history_rating = map(list, zip(*zipped_list))
            # hist_length = len(history_id)
            # cnt = 0
            # rerank_hist = []
            # rerank_rate = []
            # while len(rerank_hist) < K1 and cnt < hist_length:
            #     rerank_hist.append(history_id[cnt])
            #     rerank_rate.append(history_rating[cnt])
            #     cnt += 1
            input_dict["user_hist"] = input_dict["user_hist"][:K1]
            input_dict["hist_rating"] = input_dict["hist_rating"][:K1]
            hist_length = len(hist_rating_dict)
            cnt = 0
            recent_hist = []
            recent_rate = []
            while len(recent_hist) < K2 and cnt < hist_length:
                hist_id = seq_hist_dict[hist_length-1-cnt]
                if hist_id not in input_dict["user_hist"]:
                    recent_hist.append(hist_id)
                    recent_rate.append(hist_rating_dict[hist_id])
                cnt += 1
            # recent_hist.reverse()
            # recent_rate.reverse()
            input_dict["user_hist"] = input_dict["user_hist"] + recent_hist
            input_dict["hist_rating"] = input_dict["hist_rating"] + recent_rate

        input_dict["user_hist"] = list(map(lambda isbn: isbn2title[isbn], input_dict["user_hist"]))

        for i, (name, star) in enumerate(zip(input_dict["user_hist"], input_dict["hist_rating"])):
            suffix = " stars)" if star > 1 else " star)"
            input_dict["user_hist"][i] = f"{name} ({star}" + suffix

        yield get_template(input_dict, temp_type)


def book_hybrid_ret_get_prompt(
    K=15,
    temp_type="hybrid", # sequential, high, rerank
    data_dir="../data/BookCrossing/proc_data", 
    istrain="test", 
    emb_type="hybrid",
    indice_dir="../embeddings",
):
    assert emb_type == "hybrid"
    global input_dict, template
    fp = f"{istrain}.parquet.gz"
    df = pd.read_parquet(os.path.join(data_dir, fp)).reset_index(drop=True)
    indice_dir_1 = f"../embeddings/BookCroosing_text_indice_{istrain}.json"
    sorted_indice_1 = json.load(open(indice_dir_1))

    indice_dir_2 = f"../embeddings/BookCroosing_colla_indice_{istrain}.json"
    sorted_indice_2 = json.load(open(indice_dir_2))

    id2book = json.load(open(os.path.join(data_dir, "id2book.json"), "r"))
    isbn2id = json.load(open(os.path.join(data_dir, "isbn2id.json"), "r"))
    isbn2title = {isbn: id2book[str(isbn2id[isbn])][1] for isbn in isbn2id.keys()}

    # fill the template
    for row_number in tqdm(list(df.index)):
        row = df.loc[row_number].to_dict()

        for key in input_dict:
            if key in row.keys(): #, "Key name error."
                input_dict[key] = row[key]
        
        hist_rating_dict = {hist: rating  for hist, rating in zip(input_dict["user_hist"], input_dict["hist_rating"])}
        input_dict["user_hist"], input_dict["hist_rating"] = [], []

        K1 = (K+1)//2
        K2 = K - K1
            
        cur_indice = sorted_indice_1[row_number]
        cnt = 0
        for index in cur_indice:
            # index = str(index)
            if index in hist_rating_dict:
                cnt += 1
                input_dict["user_hist"].append(index)
                input_dict["hist_rating"].append(hist_rating_dict[index])
                if cnt == K1:
                    break
        cur_indice = sorted_indice_2[row_number]
        cnt = 0
        for index in cur_indice:
            # index = str(index)
            if index in hist_rating_dict and index not in input_dict["user_hist"]:
                cnt += 1
                input_dict["user_hist"].append(index)
                input_dict["hist_rating"].append(hist_rating_dict[index])
                if cnt == K2:
                    break

        input_dict["user_hist"] = list(map(lambda isbn: isbn2title[isbn], input_dict["user_hist"]))

        for i, (name, star) in enumerate(zip(input_dict["user_hist"], input_dict["hist_rating"])):
            suffix = " stars)" if star > 1 else " star)"
            input_dict["user_hist"][i] = f"{name} ({star}" + suffix

        yield get_template(input_dict, temp_type)


def row_to_prompt(index, df, K, isbn2title, temp_type="simple"):
    global input_dict, template
    row = df.loc[index].to_dict()
    

    for key in input_dict:
        if key in row.keys(): #, "Key name error."
            input_dict[key] = row[key]

    input_dict["user_hist"] = list(map(lambda x: isbn2title[str(x)], input_dict["user_hist"]))

    input_dict["user_hist"] = input_dict["user_hist"][-K:]
    input_dict["hist_rating"] = input_dict["hist_rating"][-K:]
    input_dict["user_hist"].reverse()
    input_dict["hist_rating"] = input_dict["hist_rating"][::-1]
    for i, (name, star) in enumerate(zip(input_dict["user_hist"], input_dict["hist_rating"])):
        suffix = " stars)" if star > 1 else " star)"
        input_dict["user_hist"][i] = f"{name} ({star}" + suffix

    return get_template(input_dict, temp_type)