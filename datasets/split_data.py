'''
split the dataset into 70 profile and 30 test per user
'''

import os
import json 
from collections import defaultdict
from math import ceil
import pandas as pd

def main():
    data_sources = ['AO3', 'narrativemagazine', 'newyorker', 'Reddit', 'Storium']
    data_split_per_source = defaultdict(dict)
    test_ratio = 0.3

    output_dir = 'data_splits'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # iterate through each data source
    for source in data_sources:
        data_dir = f'{source}/selected_human'
        # iterate through each file in the data source
        for file in os.listdir(data_dir):
            user = file.split('.')[0]
            file_path = os.path.join(data_dir, file)
            with open(file_path, 'r') as f:
                data = json.load(f)
                # split the data into 70% profile and 30% test
                proposed_test_size = ceil(len(data) * test_ratio)
                test_size = proposed_test_size if proposed_test_size > 0 else 1
                profile_size = len(data) - test_size
                if profile_size > 0:
                    data_split_per_source[source][user] = {'profile': profile_size, 'test': test_size}
        
        # save the data split
        output_path = os.path.join(output_dir, f'{source}.json')
        with open(output_path, 'w') as f:
            json.dump(data_split_per_source[source], f, indent=4)
    
    # aggregate the data split across all sources
    data_split_stats = list()
    for source in data_sources:
        profile_size = 0
        test_size = 0
        for user, split in data_split_per_source[source].items():
            profile_size += split['profile']
            test_size += split['test']
        data_split_stats.append({'Source': source, 'Profile size': profile_size, 'Test size': test_size}) 
    
    # calculate the total data split for all sources

    total_profile_size = 0
    total_test_size = 0
    for row in data_split_stats:
        total_profile_size += row['Profile size']
        total_test_size += row['Test size']
    data_split_stats.append({'Source': 'Total', 'Profile size': total_profile_size, 'Test size': total_test_size})
    
    # save the data split stats    
    df = pd.DataFrame(data_split_stats)
    df.to_csv(os.path.join(output_dir, 'data_split_stats.csv'), index=False)

    

if __name__ == "__main__":
    main()