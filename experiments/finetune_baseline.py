'''
Code for finetuning a baseline model on a dataset.
'''

import os
import json

import pandas as pd
from transformers import PreTrainedTokenizer, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset

def load_data(split='profile'):
    '''
    returns a pandas dataframe with the data (source, writing prompt, story)
    '''
    # check if data already exists
    if os.path.exists(f'../datasets/finetune_data/{split}.csv'):
        # load the data
        finetune_df = pd.read_csv(f'../datasets/finetune_data/{split}.csv')
        return finetune_df

    # define sources
    sources = ["Reddit", "AO3", "Storium", "narrativemagazine", "newyorker"]
    
    # store list
    finetune_data = []

    # iterate over sources
    for source in sources:
        split_dir = f'../datasets/data_splits/data/{source}/{split}'
        story_dir = f'../datasets/{source}/selected_human_with_prompts/'


        # iterate over files in split_dir 
        for file in os.listdir(split_dir):
            if file.endswith('.json'):
                # load the split file
                with open(os.path.join(split_dir, file), 'r') as f:
                    data = json.load(f)
                # load the story file
                with open(os.path.join(story_dir, file), 'r') as f:
                    story_data = json.load(f)
                # create dict of story data
                story_dict = {}
                for item in story_data:
                    story_dict[item['writing_prompt']] = item['comment']
                # iterate over the items
                for item in data:
                    wp = item['writing_prompt']
                    story = story_dict[wp]
                    # create dict of data
                    finetune_sample = {
                        'source': source,
                        'writing_prompt': wp,
                        'story': story
                    }
                    # append to list
                    finetune_data.append(finetune_sample)
    
    # create dataframe
    finetune_df = pd.DataFrame(finetune_data)

    # save dataframe
    finetune_data_dir = f'../datasets/finetune_data'
    if not os.path.exists(finetune_data_dir):
        os.makedirs(finetune_data_dir)
    
    finetune_df.to_csv(f'{finetune_data_dir}/{split}.csv', index=False)

    # return dataframe
    return finetune_df

def main():
    # load the dataset
    profile_df = load_data(split='profile')
    test_df = load_data(split='test')

    
if __name__ == "__main__":
    main()