'''
get category-wise pair-wise win-rates for each method and source
'''

import os 
import argparse
import re
from collections import defaultdict
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# def create_graph(method_source_wise_results, output_dir):
#     # Specify methods to compare
#     methods_to_compare = ['delta_schema_persona', 'schema_persona']

#     save_dir = f'{output_dir}/graphs'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     # Iterate over methods
#     for method, source_data in method_source_wise_results.items():
#         if method not in methods_to_compare:
#             continue

#         for source, data in source_data.items():
#             print(source)
#             if source == 'overall.json':
#                 continue

#             # Extract categories
#             categories = list(data.keys())
            
#             # Initialize bar data
#             bar_width = 0.35
#             x = np.arange(len(categories))  # Position of categories
            
#             # Plot setup
#             fig, ax = plt.subplots(figsize=(12, 6))
            
#             for i, compare_method in enumerate(methods_to_compare):
#                 if compare_method != method:
#                     continue

#                 print('In here', method)
#                 # Get data for the method
#                 method_data = data
#                 print('method_data', method_data)

#                 # Prepare bar segments
#                 expts = [method_data[cat].get('expts', 0) for cat in categories]
#                 ties = [method_data[cat].get('Tie', 0) for cat in categories]
#                 vanilla = [method_data[cat].get('vanilla', 0) for cat in categories]
                
#                 # Bottom positions for stacked bars
#                 bottom_tie = np.array(expts)
#                 bottom_vanilla = bottom_tie + np.array(ties)

#                 # Plot bars for the method
#                 ax.bar(x + i * bar_width, expts, bar_width, label=f'{method} - expts', color='blue')
#                 ax.bar(x + i * bar_width, ties, bar_width, bottom=bottom_tie, label=f'{method} - Tie', color='orange')
#                 ax.bar(x + i * bar_width, vanilla, bar_width, bottom=bottom_vanilla, label=f'{method} - vanilla', color='green')

#             # Formatting
#             ax.set_title(f'Win Rates by Category for {source}', fontsize=16)
#             ax.set_xlabel('Category', fontsize=14)
#             ax.set_ylabel('Proportion', fontsize=14)
#             ax.set_xticks(x + bar_width / 2)
#             ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=12)
#             ax.legend(fontsize=10)

#             # Save the plot to disk
#             plt.tight_layout()
#             plt.savefig(f'{save_dir}/{method}_{source}.png')
#             plt.close()

import matplotlib.pyplot as plt
import numpy as np
import os

def create_graph(method_source_wise_results, output_dir):
    # Specify methods to compare
    methods_to_compare = ['delta_schema_persona', 'schema_persona']

    save_dir = f'{output_dir}/graphs'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Define distinct colors for methods
    method_colors = {
        'delta_schema_persona': {'expts': 'blue', 'Tie': 'orange', 'vanilla': 'green'},
        'schema_persona': {'expts': 'purple', 'Tie': 'pink', 'vanilla': 'yellow'}
    }
    
    # Iterate over sources
    for source in next(iter(method_source_wise_results.values())).keys():
        if source == 'overall.json':
            continue

        # Extract categories
        categories = list(next(iter(method_source_wise_results.values()))[source].keys())
        
        # Initialize bar data
        bar_width = 0.35
        x = np.arange(len(categories))  # Position of categories
        
        # Plot setup
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, method in enumerate(methods_to_compare):
            if method not in method_source_wise_results:
                continue

            # Get data for the source in the method
            if source not in method_source_wise_results[method]:
                continue

            method_data = method_source_wise_results[method][source]

            # Prepare bar segments
            expts = [method_data[cat].get('expts', 0) for cat in categories]
            ties = [method_data[cat].get('Tie', 0) for cat in categories]
            vanilla = [method_data[cat].get('vanilla', 0) for cat in categories]
            
            # Bottom positions for stacked bars
            bottom_tie = np.array(expts)
            bottom_vanilla = bottom_tie + np.array(ties)

            # Plot bars for the method with distinct colors
            ax.bar(x + i * bar_width, expts, bar_width, label=f'{method} - expts', color=method_colors[method]['expts'])
            ax.bar(x + i * bar_width, ties, bar_width, bottom=bottom_tie, label=f'{method} - Tie', color=method_colors[method]['Tie'])
            ax.bar(x + i * bar_width, vanilla, bar_width, bottom=bottom_vanilla, label=f'{method} - vanilla', color=method_colors[method]['vanilla'])

        # Formatting
        ax.set_title(f'Win Rates by Category for {source}', fontsize=16)
        ax.set_xlabel('Category', fontsize=14)
        ax.set_ylabel('Proportion', fontsize=14)
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=12)
        ax.legend(fontsize=10)

        # Save the plot to disk
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{source}.png')
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Consolidate results from different sources for each method')
    # llama (store_true)
    parser.add_argument('--llama', action='store_true', help='Consolidate results for llama 8B')
    # llama (store_true)
    parser.add_argument('--llama70', action='store_true', help='Consolidate results for llama 70B')
    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()

    llama = args.llama
    if llama: 
        llama_suffix = '_llama'
    elif args.llama70:
        llama_suffix = '_llama70'
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
    consider_methods = ['oracle', 'vanilla_few_shot', 'delta', 'schema', 'schema_persona', 'delta_schema', 'delta_schema_persona']
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
    
    # TODO: construct graph for each source
    create_graph(method_source_wise_results, output_dir)
    
    # write the table to a csv file
    df = pd.DataFrame(rows)
    df.to_csv(f"{output_dir}/catwise_winners{llama_suffix}.csv", index=False)
    
    print(f'Saved results to {output_dir}')

if __name__ == '__main__':
    main()