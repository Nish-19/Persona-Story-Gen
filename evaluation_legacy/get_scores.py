'''
get the evaluation scores for the generated stories
'''

import os
import json
from metrics import Metrics
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--few_shot', type=bool, default=False, help='Few shot')
    # source
    parser.add_argument('--source', type=str, default='Reddit', help='Source')
    # method choice 
    parser.add_argument('--choice', type=int, default=1, help='Choice of the method: 1. Vanilla, 2. User Profile (No Schema), 3. User Profile (Schema)')
    # verbose (store_true)
    parser.add_argument('--verbose', action='store_true', help='enable verbose mode')
    # compute_gt
    parser.add_argument('--compute_gt', action='store_true', help='compute ground truth')

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
    # verbose
    verbose = args.verbose
    # compute_gt
    compute_gt = args.compute_gt

    # suffix 
    if few_shot:
        suffix = '_few_shot'
    else:
        suffix = ''

    # root directories 
    gt_root_dir = f'../datasets/data_splits/data/{source}/test/'
    if choice == 1:
        consider_dir = f'vanilla{suffix}'
    elif choice == 2:
        consider_dir = f'no_schema'
    elif choice == 3:
        consider_dir = f'schema'
    expts_root_dir = f'../experiments/results/{consider_dir}/{source}'

    # results output directory 
    output_dir = consider_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pairs = []
    # iterate over files in the ground truth directory
    for file in os.listdir(gt_root_dir):
        # gt file path
        gt_file_path = os.path.join(gt_root_dir, file)
        # expts file path
        expts_file_path = os.path.join(expts_root_dir, file)

        try:
            # read the ground truth file
            with open(gt_file_path, 'r') as f:
                gt_data = json.load(f)
            
            # read the expts file
            with open(expts_file_path, 'r') as f:
                expts_data = json.load(f)
        except:
            print('Error reading file', file)
            continue
        
        # # assert the lengths of the ground truth and expts data
        # assert len(gt_data) == len(expts_data), 'Length mismatch between ground truth and expts data'

        # # iterate over the ground truth and expts data
        # for gt, expts in zip(gt_data, expts_data):
        #     if gt['story'] is None or expts['story'] is None:
        #         print('Skipping None', file)
        #         continue
        #     pairs.append((gt['story'], expts['story']))
        
        # iterrate only over expts_data 
        for ectr, expts in enumerate(expts_data):
            gt_story = gt_data[ectr]['story']
            if gt_story is None or expts['story'] is None:
                print('Skipping None', file)
                continue
            pairs.append((gt_story, expts['story']))
        
    # initialize metrics class
    metrics = Metrics(pairs, compute_gt=compute_gt)
    # 1. Reference-based metrics
    reference_based_metrics = {
        'Rouge': metrics.compute_rouge(),
        'BLEU': metrics.compute_bleu(),
        'Bertscore': metrics.compute_bert_score(),
        'Jaccard distance': metrics.compute_jaccard_distance()
    }

    if verbose:
        print('Computed Reference-based metrics')

    # 2. Diversity metrics
    # defined config 
    config = {
        'compression_ratio' : True,
        'homo_rougel': True, 
        'homo_bert': True, 
        'homo_bleu': True, 
        'ngram': True
    }

    diversity_metrics = metrics.diversity_scores(config, verbose=verbose)
    if verbose:
        print('Computed Diversity Metrics')

    # 3. Style metrics 
    style_metrics = {
        'length': metrics.length_scores(), 
        'reading_score': metrics.flesch_reading_scores(),
        'luar_score': metrics.luar_score()
    }

    if verbose:
        print('Computed Style Metric')

    # group results 
    all_results = {
        'reference_based_metrics': reference_based_metrics, 
        'diversity_metrics': diversity_metrics, 
        'style_metrics': style_metrics
    }
    
    # save results 
    output_file = f"{output_dir}/{source}.json"
    with open(output_file, "w") as outfile:
        json.dump(all_results, outfile, indent=4)

if __name__ == '__main__':
    main()