'''
consolidate the results of user sheet prompting
'''

import os 
import re
import json
import argparse
from collections import Counter, defaultdict 

def extract_winner(res):
    '''
    extract text between the tag <winner></winner>
    '''

    score_match = re.search(r'<score>(.*?)</score>', res, re.DOTALL)
    if score_match:
        score_text = score_match.group(1)
        # Extract scores for Story A and Story B using regex
        story_a_score = re.search(r'Story A:\s*(\d+)', score_text)
        story_b_score = re.search(r'Story B:\s*(\d+)', score_text)
        
        if story_a_score and story_b_score:
            score_a = int(story_a_score.group(1).strip())
            score_b = int(story_b_score.group(1).strip())

            if score_a > score_b:
                winner = 'A'
            elif score_a < score_b:
                winner = 'B'
            else:
                winner = 'Tie'
        
            return winner

        else:
            return None
    else:
        return None

def extract_score(res):
    '''
    extract text between the tag <winner></winner>
    '''

    score_match = re.search(r'<score>(.*?)</score>', res, re.DOTALL)
    if score_match:
        score_text = score_match.group(1)
        # Extract scores for Story A and Story B using regex
        story_a_score = re.search(r'Story A:\s*(\d+)', score_text)
        story_b_score = re.search(r'Story B:\s*(\d+)', score_text)
        
        if story_a_score and story_b_score:
            score_a = int(story_a_score.group(1).strip())
            score_b = int(story_b_score.group(1).strip())

            # if score_a > score_b:
            #     winner = 'A'
            # elif score_a < score_b:
            #     winner = 'B'
            # else:
            #     winner = 'Tie'
        
            return score_a, score_b

        else:
            return None, None
    else:
        return None, None
    
def store_label_count(all_results, output_dir, suffix=''):
    '''
    store the label count
    '''
    # calculate count
    labels_count = Counter(all_results)

    # sort the labels count
    labels_count = dict(sorted(labels_count.items(), key=lambda x: x[1], reverse=True))

    labels_output_path = os.path.join(output_dir, f'winner_stats{suffix}.json')
    with open(labels_output_path, 'w') as f:
        json.dump(labels_count, f, indent=4)
    print(labels_count)


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
    parser.add_argument('--model_choice', type=int, default=1, help='Choice of the Model: 1. GPT-4o, 2. LLama-3.1-70B')

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
    # # choice
    choice = args.choice
    # model choice 
    model_choice = args.model_choice

    # choice = 3

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
    elif choice == 5:
        consider_dir = f'delta_schema{top_k_suffix}'

    if source == 'all':
        sources = ['Reddit', 'AO3', 'narrativemagazine', 'newyorker', 'Storium']
    else:
        sources = [source]

    # iterate over the sources
    for source in sources:
        eval_dir_name = 'author_sheet_score'

        eval_path = f'{eval_dir_name}/{consider_dir}/{model_choice}/{source}.json' 

        output_dir = f'{eval_dir_name}_stats/{consider_dir}/{model_choice}/{source}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


        # read the evaluation file
        with open(eval_path, 'r') as f:
            eval_data = json.load(f)
        
        # iterate over the evaluation data
        all_results = []
        all_results_score = []
        count_None = 0
        tot_count = 0
        for key, data in eval_data.items():
            category_winners = {}

            # iterate over the categories
            tot_score = defaultdict(int)
            for cat, res in data.items():
                tot_count += 1

                # get labels for A and B
                label_a = res["2"].strip("A: ")
                if label_a == 'vanilla':
                    label_b = 'expts'
                else:
                    label_b = 'vanilla'

                winner_label = extract_winner(res['1'])

                # check if winner_label is None
                if winner_label is None:
                    count_None += 1
                    # print(key, cat)
                    continue

                

                if winner_label == 'A':
                    winner = label_a
                elif winner_label == 'B':
                    winner = label_b
                else:                
                    winner = 'Tie'

                category_winners[cat] = winner

                # extract scores
                score_a, score_b = extract_score(res['1'])
                tot_score[label_a] += score_a
                tot_score[label_b] += score_b

            # overall winner
            overall_winner = Counter(category_winners.values()).most_common(1)[0][0]
            if overall_winner is None:
                continue

            # # append the overall winner
            # all_results.append(overall_winner)

            # overall winner score 
            if tot_score[label_a] > tot_score[label_b]:
                overall_winner_score = label_a
            elif tot_score[label_a] < tot_score[label_b]:
                overall_winner_score = label_b
            else:
                overall_winner_score = 'Tie'

        
            all_results_score.append(overall_winner_score)
        
        # print(f'None count: {count_None}')
        # print(f'Total count: {tot_count}')

    #    # common output 
    #     ouput_path = os.path.join(output_dir, f'winner.json')
    #     with open(ouput_path, 'w') as f:
    #         json.dump(all_results, f, indent=4)
    #     # store label count
    #     store_label_count(all_results, output_dir)
        
        # score output
        ouput_path = os.path.join(output_dir, f'winner_score.json')
        with open(ouput_path, 'w') as f:
            json.dump(all_results_score, f, indent=4)

        # store label count score
        print(f'Source: {source}')
        store_label_count(all_results_score, output_dir, suffix='_score')
        print('------------------------------------')


if __name__ == '__main__':
    main()
