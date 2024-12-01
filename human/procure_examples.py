'''
procure examples for human annotation
'''

import os 
import random
import re
import json 
import pandas as pd

def get_writing_sheet(writing_sheet_list):
    '''
    extract the writing sheet from the list of writing sheets
    '''
    # get the writing sheet
    writing_sheet = None
    for idx in range(len(writing_sheet_list)-1, -1, -1):
        try:
            writing_sheet_raw = writing_sheet_list[idx]
            # extract the sheet in the tags <combined_user_sheet></<combined_user_sheet>
            writing_sheet = re.search(r'<combined_user_sheet>(.*?)</combined_user_sheet>', writing_sheet_raw, re.DOTALL).group(1)
            if writing_sheet == '':
                writing_sheet = writing_sheet_raw
            break
        except:
            continue

    return writing_sheet


def main():
    # set random seed
    random.seed(37)

    sources = ['Reddit', 'AO3', 'Storium']

    root_dir = '../evaluation/user_sheet'
    consider_dir = 'schema'

    # pre-defined categories for evaluation
    categories = [
    "Story Beginning",
    "Story Ending",
    "Narrative Structure",
    "Unique Elements",
    "Engaging Themes and Imagery",
    "Use of Tropes or Clichés",
    "Main Character",
    "Setting Establishment",
    "Supporting Characters and Interactions",
    "Narrative Perspective",
    "Stylistic Elements",
    "Tone and Mood Alignment"
    ]   


    # iterate over the sources
    select_examples = {}
    meta_data = {}
    rows = []
    exp_count = 0
    for source in sources: 
        eval_file_path = f'{root_dir}/{consider_dir}/{source}.json'
        # root directories 
        gt_root_dir = f'../datasets/data_splits/data/{source}/test/'

        expts_root_dir = f'../experiments/results/{consider_dir}/{source}'

        # user writing sheet directory
        user_writing_sheet_dir = f'../experiments/user_profile/schema/{source}'

        # read the evaluation file
        with open(eval_file_path, 'r') as f:
            eval_data = json.load(f)
        
        # randomly select 10 examples in eval_data
        eval_keys = list(eval_data.keys())
        random.shuffle(eval_keys)
        examples = eval_keys[:15]

        # iterate over the examples
        for ectr, example in enumerate(examples): 
            # break if 10 examples are selected
            if ectr == 10:
                break


            # get file name and index 
            file, index = example.split('.json_')
            index = int(index)

            # read the ground truth file
            gt_file_path = f'{gt_root_dir}/{file}.json'
            with open(gt_file_path, 'r') as f:
                gt_data = json.load(f)
            
            # read the experiment file
            expt_file_path = f'{expts_root_dir}/{file}.json'
            with open(expt_file_path, 'r') as f:
                expt_data = json.load(f)
            
            # vanilla file path
            vanilla_file_path = f'../experiments/results/vanilla/{source}/{file}.json'
            # read the vanilla file
            with open(vanilla_file_path, 'r') as f:
                vanilla_data = json.load(f)

            
            # read the user writing sheet
            user_writing_sheet_path = f'{user_writing_sheet_dir}/{file}.json'
            with open(user_writing_sheet_path, 'r') as f:
                writing_sheet_list = json.load(f)
            
            writing_sheet = get_writing_sheet(writing_sheet_list)

            if writing_sheet is None:
                continue
            
            # choose random number - 0 or 1
            choice = random.choice([0, 1])

            if choice == 0:
                consider_dict = {
                    'Writing Prompt': gt_data[index]['writing_prompt'],
                    'User Writing Sheet': writing_sheet,
                    'Story A': vanilla_data[index]['story'],
                    'Story B': expt_data[index]['story']
                }

            else:
                consider_dict = {
                    'Writing Prompt': gt_data[index]['writing_prompt'],
                    'User Writing Sheet': writing_sheet,
                    'Story A': expt_data[index]['story'],
                    'Story B': vanilla_data[index]['story']
                }
            
            metadata_dict = {
                'file': file,
                'index': index,
                'choice': choice
            }


            # add the example to the list
            select_examples[exp_count+1] = consider_dict
            meta_data[exp_count+1] = metadata_dict

            # add row information 
            row_content = {'ID': exp_count+1}
            for cat in categories:
                row_content[cat] = ''
            
            # add the row to the list
            rows.append(row_content)

            exp_count += 1

    output_dir = f'examples/'
    os.makedirs(output_dir, exist_ok=True)

    # write the examples to a file
    with open(f'{output_dir}/data.json', 'w') as f:
        json.dump(select_examples, f, indent=4)
    
    with open(f'{output_dir}/metadata.json', 'w') as f:
        json.dump(meta_data, f, indent=4)
    
    # write the rows to a csv file
    annotation_dir = f'{output_dir}/annotation'
    if not os.path.exists(f'{annotation_dir}'):
        os.makedirs(f'{annotation_dir}')

    df = pd.DataFrame(rows)
    df.to_csv(f'{annotation_dir}/annotator_1.csv', index=False)
    df.to_csv(f'{annotation_dir}/annotator_2.csv', index=False)

if __name__ == '__main__':
    main()