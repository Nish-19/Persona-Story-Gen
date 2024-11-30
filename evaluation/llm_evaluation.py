'''
Use LLMs for pair-wise comparison of the methods
'''

import os
import json
import argparse
from tqdm import tqdm
from prompt_llm_utils import construct_prompt_message, prompt_openai

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--few_shot', type=bool, default=False, help='Few shot')
    # source
    parser.add_argument('--source', type=str, default='Reddit', help='Source')
    # method choice 
    parser.add_argument('--choice', type=int, default=1, help='Choice of the method: 1. Vanilla, 2. User Profile (No Schema), 3. User Profile (Schema), 4. Personaized Rule Generator')
    # verbose (store_true)
    parser.add_argument('--verbose', action='store_true', help='Verbose')

    return parser.parse_args()

def construct_compare_prompt_message(gt_wp, gt_story, story_a, story_b, system_prompt, user_constraints):
    '''
    construct prompt for pair-wise comparison
    '''
    input_dict = {'Writing Prompt': gt_wp, 'Human-Written Story': gt_story, 'Assistant A': story_a, 'Assistant B': story_b}
    user_instruction = f"{json.dumps(input_dict)}"

    prompt = construct_prompt_message(system_prompt, user_instruction, user_constraints)

    return prompt


def main():
    # parse arguments
    args = parse_args()

    # few shot 
    few_shot = args.few_shot
    # source 
    source = args.source
    # choice
    choice = args.choice
    # verbose
    verbose = args.verbose

    # suffix 
    if few_shot:
        suffix = '_few_shot'
    else:
        suffix = ''

    # root directories 
    gt_root_dir = f'../datasets/data_splits/data/{source}/test/'
    if choice == 1:
        consider_dir = f'vanilla{suffix}'
    elif choice == 2:
        consider_dir = f'no_schema'
    elif choice == 3:
        consider_dir = f'schema'
    elif choice == 4:
        consider_dir = f'delta'
    expts_root_dir = f'../experiments/results/{consider_dir}/{source}'

    # results output directory 
    output_dir = f"llm_evaluation/{consider_dir}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = f'{output_dir}/{source}.json'
    # check if the file exists
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            all_responses = json.load(f)
    else:
        all_responses = {}
    
    # read prompts 
    system_prompt_path = 'instructions/system_prompt/compare.txt'
    user_constraints_path = 'instructions/user_prompt/compare.txt'

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
        except:
            if verbose:
                print('Skipping', file)
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
            gt_story = gt_data[ectr]['story']
            if gt_story is None or expts['story'] is None:
                print('Skipping None', file)
                continue
            pairs.append((identifier, gt_wp, gt_story, vanilla_data[ectr]['story'], expts['story']))
    
    print(f"Using {consider_dir} method")
    print(f"Consider {len(pairs)} pairs for comparison")
    
    # iterate over the pairs
    for pair in tqdm(pairs, desc='Pair-wise Evaluation', total=len(pairs)):
        identifier, gt_wp, gt_story, vanilla_story, expts_story = pair
        prompt = construct_compare_prompt_message(gt_wp, gt_story, vanilla_story, expts_story, system_prompt, user_constraints)
        # prompt the OpenAI model
        response = prompt_openai(prompt)
        response_dict = {1: response}

        # reverse the order of the stories
        prompt = construct_compare_prompt_message(gt_wp, gt_story, expts_story, vanilla_story, system_prompt, user_constraints)
        # prompt the OpenAI model
        response = prompt_openai(prompt)
        response_dict[2] = response

        # add the responses to the list
        all_responses[identifier] = response_dict
    
        # write the responses to a file
        with open(output_file, 'w') as f:
            json.dump(all_responses, f, indent=4)
        

if __name__ == '__main__':
    main()