'''
examples for the user_sheet_annotation module
'''

import os
import pandas as pd
import re
import json 
from ast import literal_eval
from collections import defaultdict
import argparse

def extract_writing_sheet(sheet_output, key='combined_user_sheet'):
    '''
    extract text between the tags <user_writing_sheet></user_writing_sheet>
    '''            
    sheet = re.search(rf'<{key}>(.*?)</{key}>', sheet_output, re.DOTALL).group(1)
    if not sheet:
        sheet = sheet_output
    return sheet    

def organize_user_sheet(user_sheet):
    '''
    Category-wise organization of user sheet with statement and example combined.
    '''

    # Simplified category pattern to match all category headers
    category_pattern = r"###\s*(.*?)\s*(?:$|\n)"  # Matches headers like "### Plot" or "### Creativity"
    statement_pattern = r"\d+\.\s+(.+?)\n"  # Matches the numbered statements
    example_pattern = r"- Example: (.+?) \[(\d+(?:, \d+)*)\]"  # Matches the examples and sources

    categories = re.findall(category_pattern, user_sheet)  # Extract headers
    category_dict = {}

    # Split the user_sheet into sections based on categories
    sections = re.split(category_pattern, user_sheet)

    # Iterate through sections and extract statements and examples
    for i in range(1, len(sections), 2):  # Skip irrelevant parts
        category = sections[i].strip().strip("*")  # Current category
        content = sections[i + 1]  # Content for the category
        if category not in category_dict:
            category_dict[category] = []

        # Match statements and corresponding examples
        statements = re.findall(statement_pattern, content)
        examples = re.findall(example_pattern, content)

        for statement, (example, sources) in zip(statements, examples):
            category_dict[category].append({
                "statement": statement.strip().strip('"').strip("*"),  # Explicitly strip surrounding quotes
                "example": example.strip().strip("*"),
                "sources": literal_eval(f"[{sources}]")
            })

    # # Output the grouped dictionary
    # for category, items in category_dict.items():
    #     print(f"Category: {category}")
    #     for item in items:
    #         print(f"  Statement: {item['statement']}")
    #         print(f"  Example: {item['example']}")
    #         print(f"  Sources: {item['sources']}")
    #     print()

    return category_dict

def get_source_wise_claims(category_dict):
    '''
    organize claims based on sources
    '''
    source_wise_claims = defaultdict(list)

    for cat, claims in category_dict.items():
        for claim_info in claims: 
            # sources = claim_info['sources']
            # # get the minimum source number
            # min_source = min(sources)

            claim = claim_info['statement']
            example = claim_info['example']
            source = int(claim_info['sources'][0])

            # append to the source_wise_claims
            source_wise_claims[source].append((claim_info['statement'], claim_info['example'], cat))
    
    # sort source_wise_claims based on length of claims
    source_wise_claims = dict(sorted(source_wise_claims.items(), key=lambda x: len(x[1]), reverse=True))

    return source_wise_claims


def dump_annotation_sample(source_wise_claims, file, source='Reddit', threshold=3):
    '''
    construct annotation data sample
    '''
    # consider top 3 sources
    top_sources = list(source_wise_claims.keys())[:3]

    # profile story 
    profile_story_path = f'../../datasets/data_splits/data/{source}/profile/{file}'
    with open(profile_story_path, 'r') as f:
        profile_story = json.load(f)
    
    # output annotation dir 
    
    annotation_dir = f'annotation_data/{source}'
    if not os.path.exists(annotation_dir):
        os.makedirs(annotation_dir)
    
    # annotation file dir 
    annotation_file_dir = f'{annotation_dir}/{file.split('.')[0]}'
    if not os.path.exists(annotation_file_dir):
        os.makedirs(annotation_file_dir)
    
    rows = []

    # iterate over top sources
    for source in top_sources:
        # number of claims should be atleast 3
        if len(source_wise_claims[source]) < threshold:
            continue

        # get source wp and story 
        source_story = (
            profile_story[source-1]["story"]
            .encode("latin1", errors="ignore")  # Treat as Latin-1, ignoring invalid bytes
            .decode("utf-8", errors="ignore")  # Decode to UTF-8, ignoring undecodable bytes
        )

        source_wp = profile_story[source-1]["writing_prompt"]
        claims = source_wise_claims[source]

        # iterate over claims
        for claim in claims:
            rows.append({
                "source": source,
                "category": claim[2],
                "claim": claim[0],
                "example": claim[1],
                "coherence": '',
                "groundedness": '',
                "evidence": '',
                "comments": '',

            })
        
        # write source wp and story to a file
        source_info = f"#### Writing Prompt ####\n{source_wp}\n\n\n#### Story ####\n{source_story}"
        with open(f'{annotation_file_dir}/{source}.txt', 'w') as f:
            f.write(source_info)
    
    # write annotation data to a csv file
    df = pd.DataFrame(rows)
    df.to_csv(f'{annotation_file_dir}/annotation_sheet.csv', index=False)

def parse_args():
    '''
    Parse command line arguments
    '''
    parser = argparse.ArgumentParser(description='Extract examples for annotation')
    parser.add_argument('--source', type=str, default='Reddit', help='Source: Reddit, AO3, Storium, narrativemagazine, newyorker')
    # threshold (int)
    parser.add_argument('--threshold', type=int, default=3, help='Threshold for number of claims')
    args = parser.parse_args()
    return args

def main():
    # parse arguments
    args = parse_args()

    source = args.source
    # source = 'Storium'
    threshold = args.threshold

    print('Source:', source)
    print('Threshold:', threshold)

    user_sheet_dir = f'../../experiments/user_profile/delta_schema/{source}/'

    organize_sheet_dir = f'clean_user_sheet/{source}'
    if not os.path.exists(organize_sheet_dir):
        os.makedirs(organize_sheet_dir)
    
    source_wise_claims_dir = f'source_wise_claims/{source}'
    if not os.path.exists(source_wise_claims_dir):
        os.makedirs(source_wise_claims_dir)

    # iterate over all users 
    for file in os.listdir(user_sheet_dir):
        
        if file.endswith('.json'):
            user_sheet_path = os.path.join(user_sheet_dir, file)
            with open(user_sheet_path, 'r') as f:
                user_profile_list = json.load(f)
            if len(user_profile_list) == 0:
                continue
            elif len(user_profile_list) == 1:
                key = 'writing_style'
            else:
                key = 'combined_user_sheet'
            user_sheet = extract_writing_sheet(user_profile_list[-1], key)
            category_dict = organize_user_sheet(user_sheet)
            with open(f'{organize_sheet_dir}/{file}', 'w') as f:
                json.dump(category_dict, f, indent=4)
            # source wise claims
            source_wise_claims = get_source_wise_claims(category_dict)
            with open(f'{source_wise_claims_dir}/{file}', 'w') as f:
                json.dump(source_wise_claims, f, indent=4)
            # construct annotation data sample
            dump_annotation_sample(source_wise_claims, file, source, threshold)

if __name__ == '__main__':
    main()