'''
get category-wise pair-wise win-rates for each method and source
'''

import os 
import argparse
import re
from collections import defaultdict
import json
import pandas as pd

def extract_winner(res):
    '''
    extract text between the tag <winner></winner>
    '''

    def get_winner(score_text):
        # Extract scores for Story A and Story B using regex
        story_a_score = re.search(r'Assistant A:\s*(\d+)', score_text)
        story_b_score = re.search(r'Assistant B:\s*(\d+)', score_text)
        
        if story_a_score and story_b_score:
            score_a = int(story_a_score.group(1).strip())
            score_b = int(story_b_score.group(1).strip())

            if score_a > score_b:
                return 'A'
            elif score_a < score_b:
                return 'B'
            else:
                return 'Tie'
        
        else:
            return None
    winner = None
    score_match = re.search(r'<score>(.*?)</score>', res, re.DOTALL)
    if score_match:
        score_text = score_match.group(1)
        winner = get_winner(score_text)
    
    elif '**score**' in res:
        score_text = res.split('**score**')[1].split('**')[0]
        winner = get_winner(score_text)
    
    return winner


def get_catwise_winners(source_data):
    '''
    get average win-rate for each category for the source
    '''
    category_winners = defaultdict(dict)
    for key, data in source_data.items():
        

        # iterate over the categories
        for cat, res in data.items():
            # get labels for A and B
            label_a = res["2"].strip("A: ")
            if label_a == 'vanilla':
                label_b = 'expts'
            else:
                label_b = 'vanilla'

            winner_label = extract_winner(res['1'])

            # check if winner_label is None
            if winner_label is None:
                # count_None += 1
                # print(key, cat)
                continue

            if winner_label == 'A':
                winner = label_a
            elif winner_label == 'B':
                winner = label_b
            else:                
                winner = 'Tie'

            category_winners[cat][winner] = category_winners[cat].get(winner, 0) + 1
    
    # average win-rate for each category
    for cat, win_dict in category_winners.items():
        total = sum(win_dict.values())
        for key in win_dict:
            win_dict[key] /= total
    
    # sort category_winners based on values
    for cat, win_dict in category_winners.items():
        category_winners[cat] = {k: v for k, v in sorted(win_dict.items(), key=lambda item: item[1], reverse=True)}
    
    return category_winners

def parse_args():
    parser = argparse.ArgumentParser(description='Consolidate results from different sources for each method')
    # llama (store_true)
    parser.add_argument('--llama', action='store_true', help='Consolidate results for llama')
    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()

    llama = args.llama
    if llama: 
        llama_suffix = '_llama'
    else:
        llama_suffix = ''

    root_dir = f'llm_evaluation_shuffle_score{llama_suffix}' 
    # initialize the dictionary to store the results
    method_source_wise_results = defaultdict(dict)

    # iterate over directories in root_dir 
    for method in os.listdir(root_dir):
        # considering only GPT evaluated data
        method_path = f"{root_dir}/{method}/1"
        # iterate over sources in method_path 
        for source in os.listdir(method_path):
            source_path = f"{method_path}/{source}"
            # read source file
            with open(source_path, 'r') as f:
                source_data = json.load(f)
            catwise_winners = get_catwise_winners(source_data)
            # store the results
            method_source_wise_results[method][source] = catwise_winners
    
    # methodwise results (normalize across sources)
    consider_methods = ['oracle', 'vanilla_few_shot', 'delta', 'schema', 'delta_schema']
    categorywise_method_results = defaultdict(dict)
    for method, source_data in method_source_wise_results.items():
        # if method not in consider_methods:
        #     continue
        catwise_winners = defaultdict(dict)
        for source, catwise_winners_source in source_data.items():
            for cat, win_dict in catwise_winners_source.items():
                for key, val in win_dict.items():
                    catwise_winners[cat][key] = catwise_winners[cat].get(key, 0) + val
        
        # normalize the results
        for cat, win_dict in catwise_winners.items():
            total = sum(win_dict.values())
            for key in win_dict:
                win_dict[key] /= total
            
        # sort catwise_winners based on values
        for cat, win_dict in catwise_winners.items():
            catwise_winners[cat] = {k: v for k, v in sorted(win_dict.items(), key=lambda item: item[1], reverse=True)}
            if method in consider_methods:
                categorywise_method_results[cat][method] = f"{round(win_dict['expts']*100, 2)} + ({round(win_dict['Tie']*100, 2)})"
        
        method_source_wise_results[method]['overall.json'] = catwise_winners

    # construct rows for table
    rows = []
    for cat, method_dict in categorywise_method_results.items():
        row = {'category': cat}
        for method in consider_methods:
            row[method] = method_dict.get(method, 'NA')
        rows.append(row)
    
    # output dir 
    output_dir = f'consolidate_{root_dir}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # write the results to the output_dir
    for method, source_data in method_source_wise_results.items():
        for source, catwise_winners in source_data.items():
            output_sub_path = f"{output_dir}/{method}"
            if not os.path.exists(output_sub_path):
                os.makedirs(output_sub_path)
            # note source already has .json extension
            output_path = f"{output_sub_path}/{source}"
            with open(output_path, 'w') as f:
                json.dump(catwise_winners, f, indent=4)
    
    # write the table to a csv file
    df = pd.DataFrame(rows)
    df.to_csv(f"{output_dir}/catwise_winners{llama_suffix}.csv", index=False)
    
    print(f'Saved results to {output_dir}')

if __name__ == '__main__':
    main()