'''
evaluate the two stories based on the user writing sheet
'''

import os 
import sys
import json 
from ast import literal_eval
import argparse
from tqdm import tqdm
import time
import random
from collections import defaultdict
import re
from prompt_llm_utils import construct_prompt_message, prompt_openai, prompt_llama_router


def construct_compare_prompt_message(gt_wp, writing_sheet, cat, story_a, story_b, system_prompt, user_constraints):
    '''
    construct prompt for pair-wise comparison
    '''
    input_dict = {'Writing Prompt': gt_wp, 'Category to Evaluate': cat, 'Author Writing Sheet': str(writing_sheet[cat]), 'Story A': story_a, 'Story B': story_b}
    user_instruction = f"{json.dumps(input_dict)}"

    prompt = construct_prompt_message(system_prompt, user_instruction, user_constraints)

    return prompt

def decode_unicode_escapes(text):
    return text.encode('utf-8').decode('unicode_escape')

def organize_user_sheet(user_sheet):
    '''
    Category-wise organization of user sheet with statement and example combined.
    '''

    # Extract headers and content
    category_pattern = r"### \*\*(.+?)\*\*"  # Matches the category headers
    # category_pattern = r"### ?\*{0,2}(.+?)\*{0,2}"  # Matches the category headers
    statement_pattern = r"\d+\. \*\*(.+?)\*\*"  # Matches the numbered statements
    example_pattern = r"- Evidence: (.+?) \[(\d+(?:, \d+)*)\]"  # Matches the examples and sources

    categories = re.findall(category_pattern, user_sheet)  # Extract headers
    category_dict = {category: '' for category in categories}  # Initialize dictionary for each category

    # Split the user_sheet into sections based on categories
    sections = re.split(category_pattern, user_sheet)

    # Iterate through sections and extract statements and examples
    for i in range(1, len(sections), 2):  # Skip irrelevant parts
        category = sections[i].strip()  # Current category
        content = sections[i + 1]  # Content for the category

        # Match statements and corresponding examples
        statements = re.findall(statement_pattern, content)
        examples = re.findall(example_pattern, content)

        ctr = 1
        for statement, (example, sources) in zip(statements, examples):
            # category_dict[category].append(decode_unicode_escapes(statement.strip()))
            category_dict[category] += f"{ctr}. {decode_unicode_escapes(statement.strip())} "
            ctr += 1

    return category_dict

def sanitize_text(text):
    """
    Clean up hidden characters, excessive whitespaces, and normalize line endings.
    """
    # Normalize newlines to \n
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Remove non-breaking spaces and other invisible characters
    text = re.sub(r'[^\S\n]', ' ', text)  # Replace non-space whitespace with space

    # Strip leading/trailing whitespaces and normalize multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def clear_evidence(user_sheet):
    """
    Remove all 'Evidence' fields from the user_sheet and clean up the text.
    Split the cleaned claims into a list of individual claims.
    """
    # Match lines containing '- Evidence:' and remove them
    cleaned_sheet = re.sub(r' - Evidence:.*?(?=(\d+\.|$))', '', user_sheet, flags=re.DOTALL)

    # Replace multiple consecutive spaces or newlines with a single newline
    cleaned_sheet = re.sub(r'\s*\n\s*', '\n', cleaned_sheet.strip())

    # replace ** with ''
    cleaned_sheet = cleaned_sheet.replace('**', '')

    return cleaned_sheet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--few_shot', type=bool, default=False, help='Few shot')
    # few shot top k (int)
    parser.add_argument('--few_shot_top_k', type=int, default=1, help='Few Shot Top K')
    # source
    parser.add_argument('--source', type=str, default='Reddit', help='Source')
    # method choice 
    parser.add_argument('--choice', type=int, default=5, help='Choice of the method: 1. Vanilla, 2. User Profile (No Schema) 3. User Profile (Schema), 4. Personaized Rule Generator, 5. User Profile (Delta), 6. Oracle')
    # model choice 
    parser.add_argument('--model_choice', type=int, default=1, help='Choice of the Model: 1. GPT-4o, 2. LLama-3.1-70B, 3. GPT-4o-mini')
    # evaluation choice 
    parser.add_argument('--eval_choice', type=int, default=2, help='Choice of the Evaluation: 1. Author Sheet, 2. Author Sheet Schema')
    # verbose (store_true)
    parser.add_argument('--verbose', action='store_true', help='Verbose')
    # verbose (store_true)
    parser.add_argument('--azure', action='store_true', help='To use azure openai')

    return parser.parse_args()

def main():
    # parse arguments
    args = parse_args()

    # few shot 
    few_shot = args.few_shot
    # few shot top k
    few_shot_top_k = args.few_shot_top_k
    # source 
    source = args.source
    # # choice
    choice = args.choice
    # model choice 
    model_choice = args.model_choice
    # eval choice
    eval_choice = args.eval_choice
    # azure
    azure = args.azure
    # verbose
    verbose = args.verbose


    # suffix 
    if few_shot:
        suffix = '_few_shot'
    else:
        suffix = ''
    
    if few_shot_top_k == 1:
        top_k_suffix = ''
    else:
        top_k_suffix = f'_{few_shot_top_k}'

    # root directories 
    gt_root_dir = f'../datasets/data_splits/data/{source}/test/'
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

    # pre-defined categories for evaluation
    categories = [
        "Plot", 
        "Creativity",
        "Development (Character and Setting)", 
        "Language Use"
    ]


    expts_root_dir = f'../experiments/results/{consider_dir}/{source}'

    # user writing sheet directory
    if eval_choice == 1:
        user_writing_sheet_dir = f'../experiments/user_profile/delta_schema/{source}'
    elif eval_choice == 2:
        user_writing_sheet_dir = f'../experiments/user_profile/schema/{source}'

    # results output directory 
    if eval_choice == 1:
        output_dir = f"author_sheet_score/{consider_dir}/{model_choice}"
    elif eval_choice == 2:
        output_dir = f"author_sheet_score_schema/{consider_dir}/{model_choice}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = f'{output_dir}/{source}.json'
    # check if the file exists
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            all_responses = json.load(f)
        # convert all_responses to defaultdict
        all_responses = defaultdict(dict, all_responses)
    else:
        all_responses = defaultdict(dict)
    
    # read prompts 
    system_prompt_path = 'instructions/system_prompt/author_sheet_score.txt'
    user_constraints_path = 'instructions/user_prompt/author_sheet_score.txt'

    # read the system prompt
    with open(system_prompt_path, 'r') as f:
        system_prompt = f.read()
    
    # read the user constraints
    with open(user_constraints_path, 'r') as f:
        user_constraints = f.read()

    pairs = []
    # iterate over files in the ground truth directory
    for file in os.listdir(gt_root_dir):
        # gt file path
        gt_file_path = os.path.join(gt_root_dir, file)
        # vanilla file path
        vanilla_file_path = os.path.join(f'../experiments/results/vanilla/{source}', file)
        # expts file path
        expts_file_path = os.path.join(expts_root_dir, file)

        # user writing sheet path
        user_writing_sheet_path = os.path.join(user_writing_sheet_dir, file)

        # read the ground truth file
        with open(gt_file_path, 'r') as f:
            gt_data = json.load(f)
        
        try:
            # read the vanilla file
            with open(vanilla_file_path, 'r') as f:
                vanilla_data = json.load(f)
            
            # read the expts file
            with open(expts_file_path, 'r') as f:
                expts_data = json.load(f)
            
            # read the user writing sheet
            with open(user_writing_sheet_path, 'r') as f:
                writing_sheet_list = json.load(f)

        except:
            if verbose:
                print('Skipping', file)
            continue
    
        # get the writing sheet
        writing_sheet_categories = {}
        if eval_choice == 1:
            if len(writing_sheet_list) == 1:
                writing_sheet = re.search(r'<writing_style>(.*?)</writing_style>', writing_sheet_list[0], re.DOTALL).group(1)
                if writing_sheet == '':
                    writing_sheet = writing_sheet_raw

                # extract elements 
                for cctr, cat in enumerate(categories):
                    # extract text between cat and categories[cctr+1]
                    # find index of ### {cat}
                    cat_idx = writing_sheet.find(f'### {cat}')
                    # find index of ### {next category}
                    if cctr == len(categories)-1:
                        next_cat_idx = len(writing_sheet)
                    else:
                        next_cat_idx = writing_sheet.find(f'### {categories[cctr+1]}')
                    
                    # extract the text
                    writing_sheet_temp = writing_sheet[cat_idx+len(f'### {cat}'):next_cat_idx]
                    
                    # Sanitize extracted text
                    writing_sheet_temp = sanitize_text(writing_sheet_temp)

                    # Clear evidence from the writing sheet
                    writing_sheet_categories[cat] = clear_evidence(writing_sheet_temp)
            else:
                writing_sheet = None
                for idx in range(len(writing_sheet_list)-1, -1, -1):
                    try:
                        writing_sheet_raw = writing_sheet_list[idx]
                        # extract the sheet in the tags <combined_author_sheet></<combined_author_sheet>
                        writing_sheet = re.search(r'<combined_author_sheet>(.*?)</combined_author_sheet>', writing_sheet_raw, re.DOTALL).group(1)
                        if writing_sheet == '':
                            writing_sheet = writing_sheet_raw
                        break
                    except:
                        continue
                if writing_sheet is None:
                    if verbose:
                        print('Skipping None', file)
                    continue
            
                writing_sheet_categories = organize_user_sheet(writing_sheet)

        elif eval_choice == 2:
            writing_sheet = re.search(r'<writing_style>(.*?)</writing_style>', writing_sheet_list[0], re.DOTALL).group(1)
            if writing_sheet == '':
                writing_sheet = writing_sheet_raw

            # extract elements 
            for cctr, cat in enumerate(categories):
                # extract text between cat and categories[cctr+1]
                # find index of ### {cat}
                cat_idx = writing_sheet.find(f'### **{cat}**')
                # find index of ### {next category}
                if cctr == len(categories)-1:
                    next_cat_idx = len(writing_sheet)
                else:
                    next_cat_idx = writing_sheet.find(f'### **{categories[cctr+1]}**')
                
                # extract the text
                writing_sheet_temp = writing_sheet[cat_idx+len(f'### **{cat}**'):next_cat_idx]
                
                # Sanitize extracted text
                writing_sheet_temp = sanitize_text(writing_sheet_temp)

                # Clear evidence from the writing sheet
                writing_sheet_categories[cat] = clear_evidence(writing_sheet_temp)
                    
        if len(writing_sheet_categories) == 0:
            if verbose:
                print('Skipping None in writing_sheet_categories', file)
            continue
        
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


            if expts['story'] is None:
                print('Skipping None', file)
                continue
            pairs.append((identifier, gt_wp, writing_sheet_categories, vanilla_data[ectr]['story'], expts['story']))
    
    print(f"Using {consider_dir} method")
    print(f"Consider {len(pairs)} pairs for comparison")
    
    # iterate over the pairs
    for pair in tqdm(pairs, desc='Pair-wise Evaluation', total=len(pairs)):
        identifier, gt_wp, w_sheet, vanilla_story, expts_story = pair

        # iterate over the categories
        for cat in categories:
        
            # generate random number (0 or 1)
            random_number = random.randint(0, 1)
            if random_number == 0:

                prompt = construct_compare_prompt_message(gt_wp, w_sheet, cat, vanilla_story, expts_story, system_prompt, user_constraints)
                # prompt the OpenAI model
                if model_choice == 1:
                    response = prompt_openai(prompt, model='gpt-4o', azure=azure)
                elif model_choice == 2:
                    response = prompt_llama_router(prompt)
                elif model_choice == 3:
                    response = prompt_openai(prompt, model='gpt-4o-mini', azure=azure)
                response_dict = {1: response, 2: 'A: vanilla', 'Category': cat} 
            else:
                # reverse the order of the stories
                prompt = construct_compare_prompt_message(gt_wp, w_sheet, cat, expts_story, vanilla_story, system_prompt, user_constraints)
                # prompt the OpenAI model
                if model_choice == 1:
                    response = prompt_openai(prompt, azure=azure)
                elif model_choice == 2:
                    response = prompt_llama_router(prompt)
                elif model_choice == 3:
                    response = prompt_openai(prompt, model='gpt-4o-mini', azure=azure)

                response_dict = {1: response, 2: 'A: expts', 'Category': cat}

            # add the responses to the dictionary
            all_responses[identifier][cat] = response_dict
        
            # write the responses to a file
            with open(output_file, 'w') as f:
                json.dump(all_responses, f, indent=4)
            
            # sleep for 10 seconds
            time.sleep(10)

        

if __name__ == '__main__':
    main()