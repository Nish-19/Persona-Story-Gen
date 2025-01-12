'''
procure examples for Upwork annotation
'''

import os 
import re
import random
import json
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    # num_annotators
    parser.add_argument('--num_annotators', type=int, default=3, help='The number of annotators to consider')
    # num_annotators
    parser.add_argument('--common', type=int, default=15, help='Number of common stories for all annotators')
    # extra
    parser.add_argument('--extra', type=int, default=10, help='Number of exclusive stories for each annotator')
    # llama (store_true)
    parser.add_argument('--llama', action='store_true', help='Whether to use Llama')
    return parser.parse_args()

def clean_text(story_text):
    '''
    make story human readable
    '''
    # Normalize carriage returns and literal `\n` first
    story_text = story_text.replace('\r', '')  # Remove carriage returns
    story_text = re.sub(r'[ \t]*\n[ \t]*', '\n', story_text)  # Normalize spaces around newlines
    story_text = story_text.replace('\\n', '\n')  # Replace escaped \n with actual newlines
    # Clean the story_text by replacing multiple newlines with a single newline
    story_text = re.sub(r'\n+', '\n', story_text).strip()
    # remove escape characters 
    story_text = story_text.replace('\\\"', '\"')
    story_text = story_text.replace('\\\\', '\\')

    return story_text



def main():
    # set random seed
    random.seed(42)

    # parse arguments
    args = parse_args()

    # num_annotators
    num_annotators = args.num_annotators
    # common
    common = args.common
    # extra
    extra = args.extra
    # llama
    llama = args.llama

    # sources 
    sources = ['Reddit', 'AO3', 'narrativemagazine', 'newyorker', 'Storium']
    # initialize annotator dictionary 
    annotator_dict = defaultdict(list)

    # processed users
    processed_users = set()
    break_flag = False
    common_break_flag = False
    source_wise_break = {source: False for source in sources}
    ctr = 0
    cur_annotator = 0
    # iterate over sources
    while not break_flag:
        for source in sources:
            if len(annotator_dict[cur_annotator])//3 >= common + extra:
                # increment cur_annotator
                cur_annotator = (cur_annotator + 1) % num_annotators
                continue
            # check if all sources have been processed
            if all(source_wise_break.values()):
                print('All sources have been processed. Breaking...')
                break_flag = True
                break

            # initialize pairs
            pairs = [] 
            # root directories 
            gt_root_dir = f'../../datasets/data_splits/data/{source}/test/'
            vanilla_dir = f'../../experiments/results/vanilla/{source}'
            delta_dir = f'../../experiments/results/delta/{source}'
            schema_dir = f'../../experiments/results/schema/{source}'
            delta_schema_dir = f'../../experiments/results/delta_schema/{source}'

            # read files in gt_root_dir
            gt_files = os.listdir(gt_root_dir)
            # randomly pick a file that is not in processed_users
            unprocessed_users = [f for f in gt_files if f not in processed_users]
            if len(unprocessed_users) == 0:
                source_wise_break[source] = True
                continue
            else:
                gt_file = random.choice(unprocessed_users)
             
            # user name
            user_name = gt_file.split('.')[0]
            # read gt_file
            with open(os.path.join(gt_root_dir, gt_file), 'r') as f:
                gt_data = json.load(f)
            
            # pick random number in the range on length of gt_data
            common_index = random.sample(range(len(gt_data)), 1)[0]

            # read vanilla file
            with open(f"{vanilla_dir}/{gt_file}", 'r') as f:
                vanilla_data = json.load(f)
            # read delta file
            with open(f"{delta_dir}/{gt_file}", 'r') as f:
                delta_data = json.load(f)
            # read schema file
            with open(f"{schema_dir}/{gt_file}", 'r') as f:
                schema_data = json.load(f)
            # read delta_schema file
            with open(f"{delta_schema_dir}/{gt_file}", 'r') as f:
                delta_schema_data = json.load(f)
            
            # read data 
            identifier = f"{source}_{user_name}_{common_index}"
            wp, gt_story = gt_data[common_index]['writing_prompt'], clean_text(gt_data[common_index]['story'])
            vanilla_story = clean_text(vanilla_data[common_index]['story'])
            delta_story = clean_text(delta_data[common_index]['story'])
            schema_story = clean_text(schema_data[common_index]['story'])
            delta_schema_story = clean_text(delta_schema_data[common_index]['story'])

            # construct pairs
            pairs.append((identifier, wp, gt_story, vanilla_story, delta_story, 'delta'))
            pairs.append((identifier, wp, gt_story, vanilla_story, schema_story, 'schema'))
            pairs.append((identifier, wp, gt_story, vanilla_story, delta_schema_story, 'delta_schema'))
            # shuffle pairs
            random.shuffle(pairs)

            if not common_break_flag:
                # extend pairs to annotator_dict for each annotator
                for i in range(num_annotators):
                    annotator_dict[i].extend(pairs)
            else:
                # just add to current annotator 
                annotator_dict[cur_annotator].extend(pairs)
                # increment cur_annotator
                cur_annotator = (cur_annotator + 1) % num_annotators
            
            # add user_name to processed_users
            processed_users.add(gt_file)

            # increment counter
            ctr += 1

            # check if common stories have been processed
            if ctr >= common:
                common_break_flag = True
        
            # check length of each annotator_dict
            break_flag = True
            for i in range(num_annotators):
                if len(annotator_dict[i])//3 < common + extra:
                    break_flag = False
                    break
        
    # save as JSON 
    pairs_headers = ['index', 'prompt', 'ground_truth', 'story_a', 'story_b', 'expt_type']
    # add the headers to the pairs
    for i in range(num_annotators):
        for j in range(len(annotator_dict[i])):
            annotator_dict[i][j] = dict(zip(pairs_headers, annotator_dict[i][j]))
    
    output_dir = 'upwork_annotation_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # save pairs for each annotator
    for i in range(num_annotators):
        with open(f'{output_dir}/annotator_{i+1}.json', 'w') as f:
            json.dump(annotator_dict[i], f, indent=4)
            print('Length of annotator', i+1, 'is', len(annotator_dict[i]))


if __name__ == '__main__':
    main()