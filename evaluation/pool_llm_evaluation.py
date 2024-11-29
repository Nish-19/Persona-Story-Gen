'''
consolidate the results of llm_prompting
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
    # replace all \n with ''
    res = res.replace('\n', '')

    winner = re.search(r'<winner>(.*?)</winner>', res, re.DOTALL)
    try:
        winner_text = winner.group(1)
        return winner_text.strip()
    except AttributeError:
        return 'Tie'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--few_shot', type=bool, default=False, help='Few shot')
    # source
    parser.add_argument('--source', type=str, default='Reddit', help='Source')
    # method choice 
    parser.add_argument('--choice', type=int, default=1, help='Choice of the method: 1. Vanilla, 2. User Profile (No Schema), 3. User Profile (Schema)')
    # verbose
    parser.add_argument('--verbose', type=bool, default=False, help='Verbose')

    return parser.parse_args()
        

def main():
    # parse arguments
    args = parse_args()

    # few shot
    few_shot = args.few_shot

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
        consider_dir = f'schema'


    eval_path = f'llm_evaluation/{consider_dir}/{source}.json' 

    # read the evaluation file
    with open(eval_path, 'r') as f:
        eval_data = json.load(f)
    
    # iterate over the evaluation data
    all_results = []
    pair_reults = []
    for key, data in eval_data.items():
        res_1 = extract_winner(data["1"])
        res_2 = extract_winner(data["2"])
        pair_reults.append((res_1, res_2))

        if 'A' in res_1 and 'B' in res_2:
            all_results.append('A')
        elif 'B' in res_1  and 'A' in res_2:
            all_results.append('B')
        else:
            all_results.append('Tie')
    
    output_dir = f'llm_evaluation_combine/{consider_dir}/{source}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ouput_path = os.path.join(output_dir, f'winner.json')
    with open(ouput_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # dump pair results
    pair_output_path = os.path.join(output_dir, 'pair.json')
    with open(pair_output_path, 'w') as f:
        json.dump(pair_reults, f, indent=4)
    
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