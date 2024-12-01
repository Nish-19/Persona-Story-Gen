'''
consolidate the results of user sheet prompting
'''

import os 
import re
import json
import argparse
from collections import Counter 

def extract_winner(res):
    '''
    extract text between the tag <winner></winner>
    '''

    winner = re.search(r'<winner_per_category>(.*?)</winner_per_category>', res, re.DOTALL)
    category_winners = {}
    overall_winner = ''
    try:
        winner_text = winner.group(1)
        # Remove the tags and split into lines
        winner_text = winner_text.replace("\\n", "\n")

        lines = winner_text.strip().split("\n")
        lines = [line for line in lines if not re.match(r"<.*?>", line)]
        # iterate over each category and extract the winner
        for line in lines:
            line = line.strip()
            try:
                category_name, winner = line.split(': ')
                category_winners[category_name.strip()] = winner.strip()
            except:
                pass
        
        try:
            # max count of winners
            winners = Counter(category_winners.values())
            # get the winner with max count
            overall_winner = winners.most_common(1)[0][0]

            return category_winners, overall_winner
        except:
            return None, None

        
    except AttributeError:
        return None, None
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--few_shot', type=bool, default=False, help='Few shot')
    # few shot top k (int)
    parser.add_argument('--few_shot_top_k', type=int, default=1, help='Few Shot Top K')

    # source
    parser.add_argument('--source', type=str, default='Reddit', help='Source')
    # method choice 
    parser.add_argument('--choice', type=int, default=1, help='Choice of the method: 1. Vanilla, 2. User Profile (No Schema), 3. User Profile (Schema)')
    # verbose
    parser.add_argument('--verbose', type=bool, default=False, help='Verbose')
    # pool method
    parser.add_argument('--pool_choice', type=int, default=1, help='Choice of the method: 1. Standard, 2. Shuffle')

    return parser.parse_args()
        

def main():
    # parse arguments
    args = parse_args()

    # few shot
    few_shot = args.few_shot
    # few shot top k
    few_shot_top_k = args.few_shot_top_k

    if few_shot_top_k == 1:
        top_k_suffix = ''
    else:
        top_k_suffix = f'_{few_shot_top_k}'


    # source
    source = args.source
    # choice
    choice = args.choice

    # suffix 
    if few_shot:
        suffix = '_few_shot'
    else:
        suffix = ''

    # root directories 
    if choice == 1:
        consider_dir = f'vanilla{suffix}'
    elif choice == 2:
        consider_dir = f'no_schema'
    elif choice == 3:
        consider_dir = f'schema{top_k_suffix}'
    elif choice == 4:
        consider_dir = f'delta{top_k_suffix}'

    llm_eval_name = 'user_sheet'

    eval_path = f'{llm_eval_name}/{consider_dir}/{source}.json' 

    output_dir = f'{llm_eval_name}_combine/{consider_dir}/{source}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # read the evaluation file
    with open(eval_path, 'r') as f:
        eval_data = json.load(f)
    
    # iterate over the evaluation data
    all_results = []
    for key, data in eval_data.items():
        category_winners, overall_winner = extract_winner(data["1"])
        if overall_winner is None:
            continue

        gt_a = data["2"].strip("A: ")
        if overall_winner == 'A':
            all_results.append(gt_a)
        else:
            if gt_a == 'vanilla':
                all_results.append('expts')
            else:
                all_results.append('vanilla')

   # common output 
    ouput_path = os.path.join(output_dir, f'winner.json')
    with open(ouput_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    # calculate count
    labels_count = Counter(all_results)

    # sort the labels count
    labels_count = dict(sorted(labels_count.items(), key=lambda x: x[1], reverse=True))

    labels_output_path = os.path.join(output_dir, 'winner_stats.json')
    with open(labels_output_path, 'w') as f:
        json.dump(labels_count, f, indent=4)
    print(labels_count)


if __name__ == '__main__':
    main()
