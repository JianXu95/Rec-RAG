import json
import fire
from vllm import LLM, SamplingParams
import os
from tqdm import trange, tqdm
import argparse


prompt_film = '''
Write a concise and informative description of the film {film_title} whose genre is {genre}, with consideration of the following aspects:

The brief plot overview (without major spoilers).
The main themes and genre.
The setting or historical context.
Key characters and the actors who portray them.
The film's director and any notable stylistic elements.
The overall tone and mood of the film.
Its cultural impact or deeper meaning, if applicable. 

Keep the description under 80 words.

Please only return the description.
'''

prompt_movie = '''
Write a concise and informative description of the movie/TV {movie_title} whose genre is {genre}, with consideration of the following aspects:

The brief plot overview (without major spoilers).
The main themes and genre.
The setting or historical context.
Key characters and the actors who portray them.
The movie's director and any notable stylistic elements.
The overall tone and mood of the movie.
Its cultural impact or deeper meaning, if applicable. 

Keep the description under 80 words.

Please only return the description.
'''

prompt_book = '''
Write a concise and informative description of the book {book_title}. ISBN of the book is {isbn}. The author of the book is {author}.
The publication year of the book is {year}. Its publisher is {publisher}. The description could include the following details:

- A brief plot overview (without major spoilers).
- The main themes and genre.
- The setting or historical context.
- Key characters and their roles.
- The author's name and any notable stylistic elements.
- The overall tone and style of the writing.
- Any cultural or literary impact, if applicable.
- Basic publication details of year and publisher. 

Keep the description under 80 words."

Please only return the description.
'''

system_prompt = '''
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: {instruction}

ASSISTANT: 
'''


def get_textual_description(args):
    data_dir = f"../data/{args.dataset}/proc_data"

    llm = LLM(model=args.model, \
              enable_lora=False, \
            #   tensor_parallel_size=args.tensor_parallel_size
              )

    sampling_params = SamplingParams(temperature=0.8, top_p=0.90, max_tokens=1024, min_tokens=10, include_stop_str_in_output=True)
    
    all_objs = {}

    # start_index = args.start_index
    # end_index = args.end_index
    # assert start_index < end_index

    if args.dataset == "ml-1m":
        movie_dict = json.load(open(os.path.join(data_dir, "movie_detail.json"), "r"))
        for i in range(1, 3953):
            key = str(i)
            if key not in movie_dict.keys():
                title, genre = "", ""
            else:
                title, genre = movie_dict[key]
            text = prompt_film.format(film_title=title, genre=genre)
            all_objs[key] = text

    elif args.dataset == "BookCrossing":
        id2book = json.load(open(os.path.join(data_dir, "id2book.json"), "r"))
        for i in trange(len(id2book)):
            key = str(i)
            isbn, title, author, year, publisher = id2book[key]
            text = prompt_book.format(book_title=title, isbn=isbn, author=author, year=year, publisher=publisher)
            all_objs[key] = text

    elif args.dataset == "amazon-movies":
        movie_dict = json.load(open(os.path.join(data_dir, "idx2movie.json"), "r"))
        for i in trange(len(movie_dict)):
            key = str(i)
            if key not in movie_dict.keys():
                title, genre = "", ""
            else:
                _, title, genre = movie_dict[key]
            text = prompt_movie.format(movie_title=title, genre=genre)
            all_objs[key] = text
    else:
        raise NotImplementedError

    decoding_objs = all_objs

    save_path = f"../data/{args.dataset}/proc_data"
    os.makedirs(save_path, exist_ok=True)
    output_file = save_path + '/{}_text.json'.format(args.dataset)

    write_objs = {}

    for i in tqdm(range(len(decoding_objs))):
        key = str(i)
        if key not in decoding_objs.keys():
            continue
        cur_obj = decoding_objs[key]
        
        #prompt = system_prompt.format(instruction=cur_obj['prompt'])
        # prompt = system_prompt.replace("{instruction}", cur_obj)
        # problem_instruction = [prompt]
        # completions = llm.generate(
        #                     problem_instruction, 
        #                     sampling_params,
        #                     # lora_request=LoRARequest("sql_adapter", 1, lora_path)
        #                     )

        messages = [
    {
        "role": "system",
        "content": "You are a friendly and helpful asistant who always responds to the query from users",
    },
    {
        "role": "user", 
        "content": cur_obj},
]
        
        # for output in completions:
        #     prompt = output.prompt
        #     generated_text = output.outputs[0].text
            
        #     textual_description = generated_text.replace('</s>','').replace('</s','')
        # write_objs.append(textual_description)

        outputs = llm.chat(messages,
                   sampling_params=sampling_params,
                   use_tqdm=False)
        for output in outputs:
            generated_text = output.outputs[0].text
            textual_description = generated_text.replace('</s>','').replace('</s','').replace('\n','').replace('\"','')

        write_objs[key] = textual_description        
        
    # fw = open(output_file,'w')
    # for cur_obj in write_objs:
    #     fw.write(json.dumps(cur_obj)+'\n')
    # fw.close()
    json.dump(write_objs, open(output_file, "w"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/mnt/cache/sichunluo2/Llama-3.1-8B-Instruct', help="")
    parser.add_argument('--output_path', type=str, default="lora-llama", help="")
    parser.add_argument('--temperature', type=float, default=0.8, help="")
    parser.add_argument('--N', type=int, default=200, help="")
    # parser.add_argument('--tensor_parallel_size', type=int, default=1, help="")
    parser.add_argument('--max_len', type=int, default=2048, help="")
    parser.add_argument('--decoding_style', type=str, default='sampling', help="")
    parser.add_argument('--num_seqs_per_iter', type=int, default=50, help='')
    # lora config
    parser.add_argument("--use_lora", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    # Here are args of prompt
    parser.add_argument("--dataset", type=str, default="ml-1m", help="ml-1m/BookCrossing/amazon-movies")
    parser.add_argument('--start_index', type=int, default=1, help="")
    parser.add_argument('--end_index', type=int, default=3953, help="")

    args = parser.parse_args()
    get_textual_description(args)