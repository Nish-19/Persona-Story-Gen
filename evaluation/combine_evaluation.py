'''
Combine evaluation from multiple judges
'''

import os 
import json 
import pandas as pd 
from collections import defaultdict, Counter
from sklearn.metrics import cohen_kappa_score



def get_overall_winner(data):
    '''
    get combined winner labels for all judges
    '''
    combined_labels = []
    judge_values = list(data.values())
    # assert that all judges have the same number of labels
    assert all(len(judge) == len(next(iter(judge_values))) for judge in judge_values), "Judges have different number of labels"
    # iterate over the judges
    for jctr in range(len(judge_values[0])):
        judge1, judge2, judge3 = judge_values[0][jctr], judge_values[1][jctr], judge_values[2][jctr]
        win_count = Counter([judge1, judge2, judge3])
        # check if all labels are different
        if len(win_count) == 3:
            most_common_label = 'Tie'
        # elif len(win_count) == 2:
        #     if 'Tie' in win_count and win_count['Tie'] == 2:
        #         # assign label that is not 'Tie'
        #         most_common_label = [label for label in win_count if label != 'Tie'][0]
        #     else:
        #         # assign label that is not 'Tie'
        #         most_common_label = win_count.most_common(1)[0][0]
        else:
            # select most common label
            most_common_label = win_count.most_common(1)[0][0]
        
        
        combined_labels.append(most_common_label)
    
    return combined_labels


def main():
    stats_dir = "llm_evaluation_shuffle_score_stats"

    run_names = [
        'oracle',
        'vanilla_few_shot',
        'delta',
        'delta_schema_persona',
        'delta_schema',
        'schema_persona',
        'schema'
    ]

    # source names
    source_names = [
        'AO3',
        'Reddit',
        'Storium',
        'narrativemagazine',
        'newyorker'
    ]

    # winner dict
    winner_dict = defaultdict(dict)

    # winner dict stats
    winner_dict_stats = defaultdict(dict)

    # judge dict
    judge_dict = {1: 'gpt4o', 2: 'llama70B', 4: 'prometheus'}

    # save dir 
    save_dir = 'llm_evaluation_shuffle_score_combined'

    # iterate over the run names
    all_gpt4o, all_prometheus = defaultdict(list), defaultdict(list)
    for run in run_names:
        save_run_dir = f"{save_dir}/{run}"
        if not os.path.exists(save_run_dir):
            os.makedirs(save_run_dir)
        # iterate over the source names
        for source in source_names: 
            save_source_dir = f"{save_run_dir}/{source}"
            if not os.path.exists(save_source_dir):
                os.makedirs(save_source_dir)
            winner_dict[run][source] = defaultdict(dict)
            winner_dict_stats[run][source] = defaultdict(dict)
            # iterate over the judges
            for judge, judge_val in judge_dict.items():
                # winner list path
                winner_list_path = f"{stats_dir}/{run}/{judge}/{source}/winner_score.json"
                with open(winner_list_path, 'r') as f:
                    winner_list = json.load(f)
                # add to the winner dict
                winner_dict[run][source][judge_val] = winner_list
                # add gpt4o and prometheus to the all lists
                if judge == 1:
                    all_gpt4o[run].extend(winner_list)
                elif judge == 4:
                    all_prometheus[run].extend(winner_list)
                # store stats (Count)
                winner_dict_stats[run][source][judge_val] = Counter(winner_list)
                # Reorder the dict to be in the order 'expts', 'vanilla', and 'Tie'
                winner_dict_stats[run][source][judge_val] = {key: winner_dict_stats[run][source][judge_val][key] for key in ['expts', 'vanilla', 'Tie'] if key in winner_dict_stats[run][source][judge_val]}
            # get overall winner
            try:
                winner_dict[run][source]['combined'] = get_overall_winner(winner_dict[run][source])
                # store stats
                winner_dict_stats[run][source]['combined'] = Counter(winner_dict[run][source]['combined'])
            except AssertionError:
                print(f"Judges have different number of labels for {run} {source}")
                continue

            # create dataframe of winners for this run and source
            df = pd.DataFrame(winner_dict[run][source]) 
            # save the dataframe
            save_path = f"{save_source_dir}/winners.csv"
            df.to_csv(save_path, index=False)

            # save the stats
            stats_df = pd.DataFrame(winner_dict_stats[run][source])
            stats_df.insert(0, 'Label', ['expts', 'vanilla', 'Tie'])
            # save the stats
            stats_save_path = f"{save_source_dir}/stats.csv"
            stats_df.to_csv(stats_save_path, index=False)
            
    
        # calculate cohen kappa score for gpt4o and prometheus
        print(f'Run: {run}')
        kappa_score = cohen_kappa_score(all_gpt4o[run], all_prometheus[run])
        print(f'Cohen Kappa between gpt4o and prometheus:', round(kappa_score, 4))
    
    print(f'Run ALL')
    # Combine all lists from all runs into one
    combined_gpt4o = [item for sublist in all_gpt4o.values() for item in sublist]
    combined_prometheus = [item for sublist in all_prometheus.values() for item in sublist]

    # Calculate Cohen Kappa score for the combined lists
    kappa_score = cohen_kappa_score(combined_gpt4o, combined_prometheus)
    print(f'Cohen Kappa for combined gpt4o and prometheus:', round(kappa_score, 4))

if __name__ == "__main__":
    main()