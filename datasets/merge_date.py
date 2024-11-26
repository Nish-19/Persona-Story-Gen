'''
add date information to the expert stories
'''

import os 
import json
import pandas as pd 

def get_date_slots_dict(date_slots):
    '''
    convert list to dictionary
    '''

    date_slots_dict = {}

    # iterate over the slots
    for slot in date_slots:
        post_title = slot['post_title']
        date = slot['date']
        date_slots_dict[post_title] = date
    
    return date_slots_dict


def write_date_slots(root_dir):
    '''
    update the human slots with date information
    '''
    root_dir = 'narrativemagazine' 
    date_slots_dir = f"{root_dir}/date_slots"

    # iterate over the directories
    for file in os.listdir(date_slots_dir):
        file_path = os.path.join(date_slots_dir, file)
        # read the file
        with open(file_path, 'r') as f:
            date_slots = json.load(f)
        
        # read selected_human 
        with open(f"{root_dir}/selected_human/{file}", 'r') as f:
            selected_human = json.load(f)
        
        # read selected_human_with_prompts
        with open(f"{root_dir}/selected_human_with_prompts/{file}", 'r') as f:
            selected_human_with_prompts = json.load(f)
        
        # get the date_slots_dict
        date_slots_dict = get_date_slots_dict(date_slots)
        
        # iterate over dates 
        for i, slot in enumerate(selected_human):
            post_title = slot['post_title']
            date = date_slots_dict[post_title]
            # update the selected_human
            selected_human[i]['date'] = date
            # update the selected_human_with_prompts
            selected_human_with_prompts[i]['date'] = date
        
        # write the updated selected_human
        with open(f"{root_dir}/selected_human/{file}", 'w') as f:
            json.dump(selected_human, f, indent=4)
        
        # write the updated selected_human_with_prompts
        with open(f"{root_dir}/selected_human_with_prompts/{file}", 'w') as f:
            json.dump(selected_human_with_prompts, f, indent=4)
        
        print(f"Updated {file}")




def main():
    root_dir = 'narrativemagazine'
    write_date_slots(root_dir)


if __name__ == '__main__':
    main()