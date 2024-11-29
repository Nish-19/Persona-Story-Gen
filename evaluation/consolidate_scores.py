'''
consolidate scores across different methods
'''

import os 
import json 
from collections import defaultdict
import pandas as pd

def main():
    dirs = ['vanilla', 'vanilla_few_shot', 'no_schema', 'schema'] 
    source = 'Reddit'

    store_list = []
    store_dict_gt = {'method': 'gt'}
    # iterate over the directories
    for dir in dirs:
        # result file path 
        res_path = f'{dir}/{source}.json'
        # read the file
        with open(res_path, 'r') as f:
            data = json.load(f)

        # iterate over categories in data
        store_dict = {'method': dir}
        
        for cat, cat_data in data.items():
            # iterate over the data in the category
            for d_key, d_value in cat_data.items():
                # iterate over the keys in the data
                for key, value in d_value.items():
                    # add to dict
                    if dir == 'vanilla':
                        if '_gt' in key:
                            store_dict_gt[key.replace('_gt', '')] = value
                        else:
                            store_dict[key.replace('_gen', '')] = value
                    else:
                        if '_gt' in key:
                            continue
                        store_dict[key.replace('_gen', '')] = value

        # add to store list
        store_list.append(store_dict)
    
    # add gt to store list
    store_list.append(store_dict_gt)
        
    # convert to dataframe
    df = pd.DataFrame(store_list)
    output_dir = 'consolidated'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = f'{output_dir}/{source}.csv'
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    main()