'''
analyze topics of writing prompts
'''

'''
Generate writing prompts for stories
'''

import os 
import time
from tqdm import tqdm
import json 
import argparse
from prompt_llm_utils import construct_prompt_message, prompt_openai

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=int, default=1, help='Data Category. 1: AO3, 2: NarrativeMagazine, 3: NewYorker, 4: Reddit, 5. Storium')
    return parser.parse_args()


def main():
    # args
    args = parse_args()

    if args.data == 1:
        data_choice = 'AO3'
    elif args.data == 2:
        data_choice = 'narrativemagazine'
    elif args.data == 3:
        data_choice = 'newyorker'
    elif args.data == 4:
        data_choice = 'Reddit'
    elif args.data == 5:
        data_choice = 'Storium'
    elif args.data == 6:
        data_choice = 'all'
    else:
        raise ValueError("Invalid data category. Chose 1, 2, 3, 4, or 5")


    # load the system prompt 
    with open('resources/analyze/system.txt', 'r') as f:
        system_prompt = f.read()
    
    # load the user prompt
    with open('resources/analyze/user.txt', 'r') as f:
        user_prompt = f.read()
    
    if data_choice == 'all':
        sources = ['AO3', 'narrativemagazine', 'newyorker', 'Reddit', 'Storium']
    else:
        sources = [data_choice]

    out_dir = 'analyzed_prompts'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # iterate over sources
    for source in sources:
        print(f'### Processing {source} ###')

        # dictionary to store outputs
        if os.path.exists(f'{out_dir}/{source}.json'):
            # read source dict 
            with open(f'{out_dir}/{source}.json', 'r') as f:
                source_dict = json.load(f)
        else:
            source_dict = {}

        # load data
        inp_dir = f'../datasets/{source}/selected_human_with_prompts'

        # iterate through each file
        for file in tqdm(os.listdir(inp_dir), desc='Analyzing Writing Prompts', total=len(os.listdir(inp_dir))):
            if file.endswith('.json'):
                file_path = os.path.join(inp_dir, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for i, info in enumerate(data):
                        identifier = f"{file.split('.')[0]}_{i}"

                        # check if identifer exists
                        if identifier in source_dict:
                            continue

                        wp = info['writing_prompt']
                        user_prompt_inp = user_prompt.replace('<INSERT PROMPT HERE>', wp)

                        # construct prompt message
                        prompt_message = construct_prompt_message(system_prompt, user_prompt_inp)
                        # prompt openai
                        completion = prompt_openai(prompt_message, max_tokens=500, temperature=0.0)

                        # store output
                        source_dict[identifier] = completion

                        # save the data after every story
                        with open(f'{out_dir}/{source}.json', 'w') as f:
                            json.dump(source_dict, f, indent=4)
                    
                    # add sleep
                    time.sleep(2)

if __name__ == "__main__":
    main()