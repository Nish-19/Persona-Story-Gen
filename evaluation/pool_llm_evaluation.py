'''
consolidate the results of llm_prompting
'''

import os 
import re
import json
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
        

def main():
    eval_path = 'llm_evaluation/no_schema/Reddit.json' 

    # read the evaluation file
    with open(eval_path, 'r') as f:
        eval_data = json.load(f)
    
    # iterate over the evaluation data
    all_results = []
    pair_reults = []
    for data in eval_data:
        res_1 = extract_winner(data["1"])
        res_2 = extract_winner(data["2"])
        pair_reults.append((res_1, res_2))

        if 'A' in res_1 and 'B' in res_2:
            all_results.append('A')
        elif 'B' in res_1  and 'A' in res_2:
            all_results.append('B')
        else:
            all_results.append('Tie')
    
    output_dir = 'llm_evaluation_combine/no_schema/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ouput_path = os.path.join(output_dir, 'Reddit.json')
    with open(ouput_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # dump pair results
    pair_output_path = os.path.join(output_dir, 'Reddit_pair.json')
    with open(pair_output_path, 'w') as f:
        json.dump(pair_reults, f, indent=4)
    
    # calculate count
    labels_count = Counter(all_results)

    # sort the labels count
    labels_count = dict(sorted(labels_count.items(), key=lambda x: x[1], reverse=True))

    print(labels_count)


if __name__ == '__main__':
    main()