'''
Use LLMs for pair-wise comparison of the methods (random shuffle)
'''

import os
import sys
import re
import json
import argparse
from tqdm import tqdm
import random
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from prompt_llm_utils import construct_prompt_message, prompt_openai, prompt_llama, prompt_llama_router

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--few_shot', type=bool, default=False, help='Few shot')
    # few shot top k (int)
    parser.add_argument('--few_shot_top_k', type=int, default=1, help='Few Shot Top K')

    # source
    parser.add_argument('--source', type=str, default='Reddit', help='Source')
    # method choice 
    parser.add_argument('--choice', type=int, default=1, help='Choice of the method: 1. Vanilla, 2. User Profile (No Schema) 3. User Profile (Schema), 4. Personaized Rule Generator, 5. User Profile (Delta), 6. Oracle')
    # model choice 
    parser.add_argument('--model_choice', type=int, default=1, help='Choice of the Model: 1. GPT-4o, 2. LLama-3.1-70B')
    # history (store_true)
    parser.add_argument('--history', action='store_true', help='Evaluate on Past History as compared to the ground truth')
    # verbose (store_true)
    parser.add_argument('--verbose', action='store_true', help='Verbose')

    return parser.parse_args()

def construct_compare_prompt_message(gt_wp, gt_story, story_a, story_b, system_prompt, user_constraints, cat, cat_value):
    '''
    construct prompt for pair-wise comparison
    '''
    # check if gt_story is dict 
    if isinstance(gt_story, dict):
        input_dict = {'Author Style Summary': gt_story, 'Writing Prompt': gt_wp, 'Assistant A': story_a, 'Assistant B': story_b}
    # elif isinstance(gt_story, list):
    #     input_dict = {'Author History': gt_story, 'New Writing Prompt': gt_wp, 'Assistant A': story_a, 'Assistant B': story_b}
    else:
        input_dict = {'Writing Prompt': gt_wp, 'Human-Written Story': gt_story, 'Assistant A': story_a, 'Assistant B': story_b}
    
    user_instruction = f"{json.dumps(input_dict)}"
    # NOTE: Replace <Fill Here> in user_instruction with cat values
    user_constraints = user_constraints.replace('<Fill Here>', f"{cat}: {cat_value}")
    prompt = construct_prompt_message(system_prompt, user_instruction, user_constraints)
    return prompt

def construct_summarize_prompt_message(history_data, system_prompt, user_constraints, cat, cat_value):
    '''
    construct prompt for pair-wise comparison
    '''
    # check if gt_story is dict 
    input_dict = {'Author History': history_data}
    
    user_instruction = f"{json.dumps(input_dict)}"
    # NOTE: Replace <Fill Here> in user_instruction with cat values
    user_constraints = user_constraints.replace('<Fill Here>', f"{cat}: {cat_value}")
    prompt = construct_prompt_message(system_prompt, user_instruction, user_constraints)
    return prompt


def get_few_shot_indices(profile_data, example, top_k=1):
    '''
    return the few shot examples
    '''
    # get most similar examples from the profile data using BM25
    profile_prompts = [p['writing_prompt'] for p in profile_data]
    query = example['writing_prompt']

    # Tokenize the prompts and query
    stop_words = set(stopwords.words('english'))
    tokenized_prompts = [[word for word in word_tokenize(prompt.lower()) if word not in stop_words] for prompt in profile_prompts]
    tokenized_query = [word for word in word_tokenize(query.lower()) if word not in stop_words]

    # Perform BM25
    bm25 = BM25Okapi(tokenized_prompts)
    scores = bm25.get_scores(tokenized_query)
    profile_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    return profile_indices


def main():
    # parse arguments
    args = parse_args()

    # set random seed
    random.seed(37)

    # few shot 
    few_shot = args.few_shot
    # few shot top k
    few_shot_top_k = args.few_shot_top_k

    # source 
    source = args.source
    # choice
    choice = args.choice
    # model choice
    model_choice = args.model_choice
    # history
    history = args.history
    # verbose
    verbose = args.verbose

    # suffix 
    if few_shot:
        suffix = '_few_shot'
    else:
        suffix = ''

    # history 
    if history:
        his_suffix = '_summarize_history'
    else:
        his_suffix = ''

    if few_shot_top_k == 1:
        top_k_suffix = ''
    else:
        top_k_suffix = f'_{few_shot_top_k}'


    # root directories 
    gt_root_dir = f'../datasets/data_splits/data/{source}/test/'
    profile_root_dir = f'../datasets/data_splits/data/{source}/profile/'
    if choice == 1:
        consider_dir = f'vanilla{suffix}'
    elif choice == 2:
        consider_dir = f'no_schema'
    elif choice == 3:
        consider_dir = f'schema{top_k_suffix}'
    elif choice == 4:
        consider_dir = f'delta{top_k_suffix}'
    elif choice == 5:
        consider_dir = f'delta_schema{top_k_suffix}'

    expts_root_dir = f'../experiments/results/{consider_dir}/{source}'

    # results output directory 
    output_dir = f"llm_evaluation_shuffle_score{his_suffix}/{consider_dir}/{model_choice}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = f'{output_dir}/{source}.json'
    # check if the file exists
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            all_responses = json.load(f)
    else:
        all_responses = {}
    
    # check history 
    if history: 
        output_summarize_history_dir = f'{output_dir}/summarize_history'
        if not os.path.exists(output_summarize_history_dir):
            os.makedirs(output_summarize_history_dir)
        output_summarize_history_path = f'{output_summarize_history_dir}/{source}.json'
        # check if the file exists
        if os.path.exists(output_summarize_history_path):
            with open(output_summarize_history_path, 'r') as f:
                all_responses_summarize_history = json.load(f)
        else:
            all_responses_summarize_history = {}
    
    # read compare prompts 
    system_prompt_path = f'instructions/system_prompt/compare_score{his_suffix}.txt'
    user_constraints_path = f'instructions/user_prompt/compare_score{his_suffix}.txt'

    # read summarize history prompts 
    system_prompt_sumhis_path = 'instructions/system_prompt/summarize_history.txt'
    user_constraints_sumhis_path = 'instructions/user_prompt/summarize_history.txt'

    categories_path = 'instructions/user_prompt/compare_categories.json'

    # NOTE: 1. Compare prompts 
    # read the system prompt
    with open(system_prompt_path, 'r') as f:
        system_prompt = f.read()
    # read the user constraints
    with open(user_constraints_path, 'r') as f:
        user_constraints = f.read()
    
    # NOTE: 2. Summarize History prompts
    # read the system prompt
    with open(system_prompt_sumhis_path, 'r') as f:
        system_prompt_sumhis = f.read()
    # read the user constraints
    with open(user_constraints_sumhis_path, 'r') as f:
        user_constraints_sumhis = f.read()
    
    # read the categories
    with open(categories_path, 'r') as f:
        categories_data = json.load(f)
    
    # define the categories
    categories = categories_data.keys()

    pairs = []
    # iterate over files in the ground truth directory
    for file in tqdm(os.listdir(gt_root_dir), desc='Processing Authors', total=len(os.listdir(gt_root_dir))):
        # gt file path
        gt_file_path = os.path.join(gt_root_dir, file)
        # profile file path
        profile_file_path = os.path.join(profile_root_dir, file)
        # vanilla file path
        vanilla_file_path = os.path.join(f'../experiments/results/vanilla/{source}', file)
        # expts file path
        expts_file_path = os.path.join(expts_root_dir, file)

        # read the ground truth file
        with open(gt_file_path, 'r') as f:
            gt_data = json.load(f)
        
        # read the profile file
        with open(profile_file_path, 'r') as f:
            profile_data = json.load(f)
        
        try:
            # read the vanilla file
            with open(vanilla_file_path, 'r') as f:
                vanilla_data = json.load(f)
            
            # read the expts file
            with open(expts_file_path, 'r') as f:
                expts_data = json.load(f)
        except:
            if verbose:
                print('Skipping', file)
            continue
    
        if history:
            # last_story_wp = profile_data[-1]['writing_prompt']
            # last_story = profile_data[-1]['story']
            # last_story_data = {'writing_prompt': last_story_wp, 'story': last_story}
            # get all the history data
            history_data = [{'writing_prompt': p['writing_prompt'], 'story': p['story']} for p in profile_data]

            # TODO: summarize history for each category
            # 1. Check if file exists in all_responses_summarize_history
            if file in all_responses_summarize_history:
                summarize_history = all_responses_summarize_history[file]
            else:
                summarize_history = {}
                for cat in categories:
                    # construct the prompt
                    prompt = construct_summarize_prompt_message(history_data, system_prompt_sumhis, user_constraints_sumhis, cat, categories_data[cat])
                    # prompt the OpenAI model
                    if model_choice == 1:
                        response = prompt_openai(prompt)
                    elif model_choice == 2:
                        response = prompt_llama_router(prompt)
                    # extract response in the tags <analysis></analysis>
                    response_match = re.search(r'<analysis>(.*)</analysis>', response, re.DOTALL)
                    if response_match:
                        response = response_match.group(1)
                    else:
                        response = response

                    summarize_history[cat] = response
                
                    # add the responses to the dictionary
                    all_responses_summarize_history[file] = summarize_history
                    # write the responses to a file
                    with open(output_summarize_history_path, 'w') as f:
                        json.dump(all_responses_summarize_history, f, indent=4)
        # else:
        #     last_story_data = None
        
        
        # iterrate only over expts_data 
        for ectr, expts in enumerate(expts_data):
            # add the pair
            identifier = f"{file}_{ectr}"

            # check if the identifier exists in the output file
            if identifier in all_responses:
                if verbose:
                    print(f"Skipping {identifier}")
                continue

            gt_wp = gt_data[ectr]['writing_prompt']
            gt_story = gt_data[ectr]['story']
            if gt_story is None or expts['story'] is None:
                print('Skipping None', file)
                continue
            if history:
                # history_data = [last_story_data]
                # # get the history data (most similar BM25)
                # profile_indices = get_few_shot_indices(profile_data, gt_data[ectr], top_k=3)
                # # profile_indices = [i for i in range(len(profile_data))]
                # if last_story_data is None:
                #     history_wp = profile_data[profile_indices[0]]['writing_prompt']
                #     history_story = profile_data[profile_indices[0]]['story']
                #     history_data = {'writing_prompt': history_wp, 'story': history_story}
                # else:
                #     # iterate over profile_indices
                #     for index in profile_indices:
                #         history_wp = profile_data[index]['writing_prompt']
                #         # check if history_wp is same as recent history data
                #         if history_wp == last_story_data['writing_prompt']:
                #             continue
                #         else:
                #             history_story = profile_data[index]['story']
                #             bm25_data = {'writing_prompt': history_wp, 'story': history_story}
                #             history_data.append(bm25_data)
                #             break
                #     # history_data = [last_story_data, bm25_data]

                pairs.append((identifier, gt_wp, summarize_history, vanilla_data[ectr]['story'], expts['story']))
            else:
                pairs.append((identifier, gt_wp, gt_story, vanilla_data[ectr]['story'], expts['story']))
        
    
    print(f"Using {consider_dir} method")
    print(f"Consider {len(pairs)} pairs for comparison")

    
    # iterate over the pairs
    for pair in tqdm(pairs, desc='Pair-wise Evaluation', total=len(pairs)):
        identifier, gt_wp, gt_story, vanilla_story, expts_story = pair

        cat_dict = {}
        for cat in categories:
            # check type of gt_story
            if isinstance(gt_story, dict):
                gt_story_input = {cat: gt_story[cat]}
            else:
                gt_story_input = gt_story

            # generate random number (0 or 1)
            random_number = random.randint(0, 1)
            if random_number == 0:

                prompt = construct_compare_prompt_message(gt_wp, gt_story_input, vanilla_story, expts_story, system_prompt, user_constraints, cat, categories_data[cat])
                # prompt the OpenAI model
                if model_choice == 1:
                    response = prompt_openai(prompt)
                elif model_choice == 2:
                    response = prompt_llama_router(prompt)
                response_dict = {1: response, 2: 'A: vanilla'} 
            else:
                # reverse the order of the stories
                prompt = construct_compare_prompt_message(gt_wp, gt_story_input, expts_story, vanilla_story, system_prompt, user_constraints, cat, categories_data[cat])
                # prompt the OpenAI model
                if model_choice == 1:
                    response = prompt_openai(prompt)
                elif model_choice == 2:
                    response = prompt_llama_router(prompt)
                response_dict = {1: response, 2: 'A: expts'}
            
            cat_dict[cat] = response_dict

            # add the responses to the list
            all_responses[identifier] = cat_dict
        
            # write the responses to a file
            with open(output_file, 'w') as f:
                json.dump(all_responses, f, indent=4)
        

if __name__ == '__main__':
    main()