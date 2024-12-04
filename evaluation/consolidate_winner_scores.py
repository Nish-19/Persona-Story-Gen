'''
group the winner scores
'''

import os 
import json 
import pandas as pd

def main():
    # root_dir = 'llm_evaluation_shuffle_combine'
    suffix = ''

    root_dir = 'user_sheet_score_combine'
    suffix = '_score'

    output_root_dir = f'consolidate_winner/{root_dir}'
    os.makedirs(output_root_dir, exist_ok=True)

    # iterate over methods in the root directory
    all_methods_win_rates = {}
    for method in os.listdir(root_dir):
        source_wise_win_rates = []
        # iterate over the sources in the method directory
        for source in os.listdir(os.path.join(root_dir, method)):
            # source path 
            source_path = os.path.join(root_dir, method, source)
            # winner stats file 
            winner_stats_file_path = os.path.join(source_path, f'winner_stats{suffix}.json')
            # read the winner stats file
            with open(winner_stats_file_path, 'r') as f:
                winner_stats = json.load(f)
            # get winning percentage
            expts_win = winner_stats['expts']
            tot_win = sum(winner_stats.values())
            win_per = round((expts_win / tot_win) * 100, 2)
            # add to the source wise win rates
            source_wise_win_rates.append({
                'source': source,
                'win_per': win_per
            })
        all_methods_win_rates[method] = source_wise_win_rates
        
        # dump the source wise win rates as csv file
        source_wise_win_rates_df = pd.DataFrame(source_wise_win_rates)
        source_wise_win_rates_df.to_csv(os.path.join(output_root_dir, f'{method}.csv'), index=False)
    
    # concatenate all source wise win rates into a single DataFrame
    all_methods_win_rates_df = pd.concat(
        [pd.DataFrame(source_wise_win_rates).assign(method=method) for method, source_wise_win_rates in all_methods_win_rates.items()],
        ignore_index=True
    )
    # Pivot the DataFrame to have methods as columns
    pivot_df = all_methods_win_rates_df.pivot(index='source', columns='method', values='win_per')

    # Reset the index to make 'source' a column again
    pivot_df.reset_index(inplace=True)
    pivot_df.to_csv(os.path.join(output_root_dir, 'all_methods.csv'), index=False)

    # all_methods_win_rates_df.to_csv(os.path.join(output_root_dir, 'all_methods.csv'), index=False)

if __name__ == '__main__':
    main()