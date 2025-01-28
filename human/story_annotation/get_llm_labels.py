'''
Get LLM labels for the annotation dataset
'''

import os 
import re
import sys
import json
import pandas as pd
from collections import defaultdict, Counter

def get_winner(data):
    '''
    get winner from the analysis
    '''
    def extract_score(res):
        '''
        extract text between the tag <winner></winner>
        '''

        def get_scores(score_text):
            # Extract scores for Story A and Story B using regex
            story_a_score = re.search(r'Assistant A:\s*(\d+)', score_text)
            story_b_score = re.search(r'Assistant B:\s*(\d+)', score_text)
            
            if story_a_score and story_b_score:
                score_a = int(story_a_score.group(1).strip())
                score_b = int(story_b_score.group(1).strip())

                return score_a, score_b
            
            else:
                return None, None

        score_match = re.search(r'<score>(.*?)</score>', res, re.DOTALL)
        if score_match:
            score_text = score_match.group(1)
            return get_scores(score_text)
        elif '**score**' in res:
            score_text = res.split('**score**')[1].split('**')[0]
            return get_scores(score_text)
        else:
            return None, None

    # iterate over the categories
    tot_score = defaultdict(int)
    for cat, res in data.items():
        # get labels for A and B
        label_a = res["2"].strip("A: ")
        if label_a == 'vanilla':
            label_b = 'expts'
        else:
            label_b = 'vanilla'

        # extract scores
        score_a, score_b = extract_score(res['1'])
        tot_score[label_a] += score_a
        tot_score[label_b] += score_b

    # overall winner score 
    if tot_score[label_a] > tot_score[label_b]:
        overall_winner_score = label_a
    elif tot_score[label_a] < tot_score[label_b]:
        overall_winner_score = label_b
    else:
        overall_winner_score = 'Tie'
    
    return overall_winner_score


def main():
    upwork_dir = 'upwork_annotation_data'
    annotator_files = [f for f in os.listdir(upwork_dir) if f.endswith('.json')]
    # read the annotation files
    annotator_data = []

    for annotator_file in annotator_files:
        with open(os.path.join(upwork_dir, annotator_file), 'r') as f:
            annotator_data.append(json.load(f))
    
    root_llm_dir = '../../evaluation/llm_evaluation_shuffle_score'

    indices, expt_types, similarity_gpt4o, similarity_llama = [], [], [], []
    winner_alias = {'expts': 'B', 'vanilla': 'A', 'Tie': 'T'}

    # get llm labels for each annotation file
    for dctr, data in enumerate(annotator_data):
        for i, d in enumerate(data):
            if dctr > 0 and i < 45:
                continue
            # TODO: get llm labels
            index_split = d['index'].split('_')
            source, story_num = index_split[0], index_split[-1]
            author_name = '_'.join(index_split[1:-1])
            story_name = f"{author_name}.json_{story_num}"
            # expt type
            expt_type = d['expt_type']
            # store results
            indices.append(d['index'])
            expt_types.append(expt_type)

            for choice in [1, 2]:
                # set evaluation path 
                eval_path = f"{root_llm_dir}/{expt_type}/{choice}/{source}.json"
                # read evaluation file
                with open(eval_path, 'r') as f:
                    eval_data = json.load(f)
                # get analysis for the story_name
                analysis = eval_data[story_name]
                # get winner
                winner = get_winner(analysis)
                if choice == 1:
                    similarity_gpt4o.append(winner_alias[winner])
                else:
                    similarity_llama.append(winner_alias[winner])
    
    # store as dataframe
    df = pd.DataFrame({'index': indices, 'expt_type': expt_types, 'similarity_gpt4o': similarity_gpt4o, 'similarity_llama': similarity_llama})
    df.to_csv('llm_labels.csv', index=False)
    print('Length of data:', len(df))

if __name__ == '__main__':
    main()