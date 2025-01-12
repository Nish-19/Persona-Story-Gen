'''
select examples for annotation
'''

import os 
import random
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--few_shot', type=bool, default=False, help='Few shot')
    # few shot top k (int)
    parser.add_argument('--few_shot_top_k', type=int, default=1, help='Few Shot Top K')
    # # source
    # parser.add_argument('--source', type=str, default='Reddit', help='Source')
    # method choice 
    parser.add_argument('--choice', type=int, default=1, help='Choice of the method: 1. Vanilla, 2. User Profile (No Schema) 3. User Profile (Schema), 4. Personaized Rule Generator, 5. User Profile (Delta), 6. Oracle')
    # verbose (store_true)
    parser.add_argument('--verbose', action='store_true', help='Verbose')

    return parser.parse_args()

def main():
    # parse arguments
    args = parse_args()

    # set random seed
    random.seed(37)

    # few shot 
    few_shot = args.few_shot
    # few shot top k
    few_shot_top_k = args.few_shot_top_k

    # # source 
    # source = args.source
    # choice
    choice = args.choice
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

    # sources 
    sources = ['Reddit', 'AO3', 'narrativemagazine', 'newyorker', 'Storium']

    # iterate over sources
    pairs = []
    for source in sources: 
        print(f"Processing {source}")
        # root directories 
        gt_root_dir = f'../../datasets/data_splits/data/{source}/test/'
        profile_root_dir = f'../../datasets/data_splits/data/{source}/profile/'
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

        expts_root_dir = f'../../experiments/results/{consider_dir}/{source}'
        
        # # read prompts 
        # categories_path = 'instructions/user_prompt/compare_categories.json'
        
        # # read the categories
        # with open(categories_path, 'r') as f:
        #     categories_data = json.load(f)

        # iterate over files in the ground truth directory
        for file in os.listdir(expts_root_dir):
            # gt file path
            gt_file_path = os.path.join(gt_root_dir, file)
            # profile file path
            profile_file_path = os.path.join(profile_root_dir, file)
            # vanilla file path
            vanilla_file_path = os.path.join(f'../../experiments/results/vanilla/{source}', file)
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
                    print('Skipping', source, file)
                continue
        
            # choose random index in expts_data
            rindex = random.randint(0, len(expts_data) - 1)
            identifier = f"{file}_{rindex}"
            try:
                # choose random between 0 and 1
                data_choice = random.choice([0, 1])
                if data_choice == 0:
                    pairs.append((identifier, gt_data[rindex]['writing_prompt'], gt_data[rindex]['story'], vanilla_data[rindex]['story'], expts_data[rindex]['story'], 'vanilla'))
                else:
                    pairs.append((identifier, gt_data[rindex]['writing_prompt'], gt_data[rindex]['story'], expts_data[rindex]['story'], vanilla_data[rindex]['story'], 'expts'))
            except IndexError:
                if verbose:
                    print('Skipping', file)
                continue
                        
    print(f"Using {consider_dir} method")
    print(f"Consider {len(pairs)} pairs for comparison")

    # save as JSON 
    pairs_headers = ['index', 'prompt', 'ground_truth', 'story_a', 'story_b', 'first_choice'] 
    # add the headers to the pairs 
    pair_list = [{header: value for header, value in zip(pairs_headers, pair)} for pair in pairs]

    output_dir = 'sample_annotation'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open('sample_annotation/pairs.json', 'w') as f:
        json.dump(pair_list, f, indent=4)

if __name__ == '__main__':
    main()
