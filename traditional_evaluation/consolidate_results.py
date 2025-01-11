'''
group results 
'''

import os 
import json 
from collections import defaultdict
import pandas as pd 

def get_consolidated_results(res_dir, store_gt=False):
    # initialie consolidated results
    consolidate_results, consolidate_results_gt = {}, {}

    #  iterate over directories in res_dir
    for sctr, source_dir in enumerate(os.listdir(res_dir)):
        source_dir_path = os.path.join(res_dir, source_dir)
        if os.path.isdir(source_dir_path):
            # iterate over files in the source directory
            for file in os.listdir(source_dir_path):
                file_path = os.path.join(source_dir_path, file)
                if os.path.isfile(file_path):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # itearte over data
                        for overall_cat, cat_results in data.items():
                            for cat, cat_result in cat_results.items():
                                if cat == 'precision' or cat == 'recall':
                                    continue
                                if cat == 'f1':
                                    cat = f'bertscore_{cat}'
                                if store_gt:
                                    if '_gt' in cat:
                                        new_cat = cat.replace('_gt', '')
                                        if sctr == 0:
                                            consolidate_results_gt[new_cat] = cat_result
                                        else: 
                                            if new_cat in consolidate_results_gt:
                                                consolidate_results_gt[new_cat] += cat_result
                                    elif '_gen' not in cat:
                                        consolidate_results_gt[cat] = ''
                                if '_gt' not in cat:
                                    new_cat = cat.replace('_gen', '')
                                    if sctr == 0:
                                        consolidate_results[new_cat] = cat_result
                                    else:
                                        if new_cat in consolidate_results:
                                            consolidate_results[new_cat] += cat_result
    
    # average the results
    num_results = len(os.listdir(res_dir))
    for key in consolidate_results:
        consolidate_results[key] /= num_results
    for key in consolidate_results_gt:
        if consolidate_results_gt[key] != '':
            consolidate_results_gt[key] /= num_results
    
    return consolidate_results, consolidate_results_gt


def main():
    # root_dir = 'results'
    root_dir = 'results_llama'
    vanilla_dir = f"{root_dir}/vanilla"
    consolidate_vanilla, consolidate_gt = get_consolidated_results(vanilla_dir, store_gt=True)
    assert len(consolidate_vanilla) == len(consolidate_gt)

    # intialize rows 
    rows = []
    # NOTE: add consolidate_gt to rows
    consolidate_gt['method'] = 'ground_truth'
    # reverse the order of dictionary keys
    consolidate_gt = dict(reversed(list(consolidate_gt.items())))
    rows.append(consolidate_gt)


    # iterate over directories in root_dir
    source_dirs = ['oracle', 'vanilla', 'vanilla_few_shot', 'delta', 'schema', 'delta_schema']
    for source_dir in source_dirs:
        if source_dir == 'vanilla':
            # NOTE: add consolidate_vanilla to rows
            consolidate_vanilla['method'] = 'vanilla'
            # reverse the order of dictionary keys
            consolidate_vanilla = dict(reversed(list(consolidate_vanilla.items())))
            rows.append(consolidate_vanilla)
        else:
            source_dir_path = os.path.join(root_dir, source_dir)
            if os.path.isdir(source_dir_path):
                consolidate_results, _ = get_consolidated_results(source_dir_path, store_gt=False)
                consolidate_results['method'] = source_dir
                # reverse the order of dictionary keys
                consolidate_results = dict(reversed(list(consolidate_results.items())))
                rows.append(consolidate_results)
    
    # create a dataframe
    df = pd.DataFrame(rows)
    # round all float rows to two decimal places
    # df = df.round(4)
    df = df.apply(lambda col: col.map(lambda x: f"{x:.4f}" if isinstance(x, float) else x))
    # save the dataframe
    df.to_csv(f'consolidated_{root_dir}.csv', index=False)

    print(f'Saved results to consolidated_{root_dir}.csv')

if __name__ == '__main__':
    main()