'''
get the evaluation scores for the generated stories
'''

import os
import json
from metrics import Metrics
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # few shot
    parser.add_argument('--few_shot', action='store_true', help='Few Shot Story Generation')
    # few shot top k (int)
    parser.add_argument('--few_shot_top_k', type=int, default=1, help='Few Shot Top K')
    # source
    parser.add_argument('--source', type=str, default='Reddit', help='Source')
    # method choice 
    parser.add_argument('--choice', type=int, default=1, help='Choice of the method: 1. Vanilla, 2. User Profile (No Schema) 3. User Profile (Schema), 4. Personaized Rule Generator, 5. User Profile (Delta), 6. Oracle')
    # verbose (store_true)
    parser.add_argument('--verbose', action='store_true', help='Verbose')
    # llama (store_true)
    parser.add_argument('--llama', action='store_true', help='To use llama generated model results')
    # compute_gt (store_true)
    parser.add_argument('--compute_gt', action='store_true', help='Compute Ground Truth')
    # persona mode
    parser.add_argument('--persona', action='store_true', help='To use persona prompt obtained from Author Sheet (for Schema and Delta Schema only)')


    return parser.parse_args()

def main():
    # parse arguments
    args = parse_args()

    # few shot 
    few_shot = args.few_shot
    # few shot top k
    few_shot_top_k = args.few_shot_top_k
    # source 
    source = args.source
    # choice
    choice = args.choice
    # verbose
    verbose = args.verbose
    # compute_gt
    compute_gt = args.compute_gt
    # llama 
    llama = args.llama
    # persona
    persona = args.persona

    # suffix 
    if few_shot:
        suffix = '_few_shot'
    else:
        suffix = ''

    if few_shot_top_k == 1:
        top_k_suffix = ''
    else:
        top_k_suffix = f'_{few_shot_top_k}'

    # llama_suffix
    if llama:
        llama_suffix = '_llama'
    else:
        llama_suffix = ''

    # persona suffix
    if persona:
        persona_suffix = '_persona'
    else:
        persona_suffix = ''


    # root directories 
    if choice == 1:
        consider_dir = f'vanilla{suffix}'
    elif choice == 2:
        consider_dir = f'no_schema'
    elif choice == 3:
        consider_dir = f'schema{top_k_suffix}{persona_suffix}'
    elif choice == 4:
        consider_dir = f'delta{top_k_suffix}'
    elif choice == 5:
        consider_dir = f'delta_schema{top_k_suffix}{persona_suffix}'
    elif choice == 6:
        consider_dir = f'oracle{top_k_suffix}'
    
    if source == 'all':
        sources = ['Reddit', 'AO3', 'Storium', 'narrativemagazine', 'newyorker']
    else:
        sources = [source]
    
    # iterate over sources 
    for source in sources: 
        print(f'### Processing {source} ###')
        # ground truth directory
        gt_root_dir = f'../datasets/data_splits/data/{source}/test/'

        expts_root_dir = f'../experiments/results{llama_suffix}/{consider_dir}/{source}'

        # results output directory 
        output_root_dir = f'results{llama_suffix}'
        output_dir = f"{output_root_dir}/{consider_dir}/{source}"
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
                try:
                    gt_story = gt_data[ectr]['story']
                except: 
                    continue
                if gt_story is None or expts['story'] is None:
                    print('Skipping None', file)
                    continue
                pairs.append((gt_story, expts['story']))
        
        print(f'## Processing {len(pairs)} pairs')
            
        # initialize metrics class
        metrics = Metrics(pairs, compute_gt=compute_gt)

        # 1. Reference-based metrics

        # check if file exists
        if os.path.exists(f'{output_dir}/reference_based_metrics.json'):
            if verbose:
                print('Reference-based metrics already computed')
        else:
            reference_based_metrics = {
                'Rouge': metrics.compute_rouge(),
                'BLEU': metrics.compute_bleu(),
                'Bertscore': metrics.compute_bert_score(),
                'Jaccard distance': metrics.compute_jaccard_distance()
            }

            if verbose:
                print('Computed Reference-based metrics')

            # save reference_based_metrics
            reference_based_metrics_path = f'{output_dir}/reference_based_metrics.json'
            with open(reference_based_metrics_path, 'w') as outfile:
                json.dump(reference_based_metrics, outfile, indent=4)

        # 2. Diversity metrics
        # defined config 
        if os.path.exists(f'{output_dir}/diversity_metrics.json'):
            if verbose:
                print('Diversity metrics already computed')
        else:
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

            # save diversity metrics
            diversity_metrics_path = f'{output_dir}/diversity_metrics.json'
            with open(diversity_metrics_path, 'w') as outfile:
                json.dump(diversity_metrics, outfile, indent=4)


        # 3. Style metrics 
        if os.path.exists(f'{output_dir}/style_metrics.json'):
            if verbose: 
                print('Style metrics already computed')
        else:
            style_metrics = {
                'length': metrics.length_scores(), 
                'reading_score': metrics.flesch_reading_scores(),
                'luar_score': metrics.luar_score()
            }

            if verbose:
                print('Computed Style Metric')

            # save style metrics 
            style_metrics_path = f'{output_dir}/style_metrics.json'
            with open(style_metrics_path, 'w') as outfile:
                json.dump(style_metrics, outfile, indent=4)

        if verbose:
            print('Saved the evaluation scores')

if __name__ == '__main__':
    main()