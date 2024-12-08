'''
examples for the user_sheet_annotation module
'''

import os
import pandas as pd
import re
import json 
from ast import literal_eval
from collections import defaultdict

def extract_writing_sheet(sheet_output, key='combined_user_sheet'):
    '''
    extract text between the tags <user_writing_sheet></user_writing_sheet>
    '''            
    sheet = re.search(rf'<{key}>(.*?)</{key}>', sheet_output, re.DOTALL).group(1)
    if not sheet:
        sheet = sheet_output
    return sheet    

# def organize_user_sheet(user_sheet):
#     '''
#     get individual statements and sources from the user sheet
#     '''
#     # Extract sentences and sources
#     pattern = r"- (.+?)\s\[(\d+(?:, \d+)*)\]"
#     matches = re.findall(pattern, user_sheet)

#     # Prepare output
#     output = [{"statement": match[0].strip(), "sources": f"[{match[1]}]"} for match in matches]

#     # Print or use the output
#     for item in output:
#         print(f"Statement: {item['statement']}\nSources: {item['sources']}\n")

def organize_user_sheet(user_sheet):
    '''
    category-wise organization of user sheet
    '''

    # Extract headers and content
    category_pattern = r"### \*\*(.+?)\*\*"  # Matches the category headers
    content_pattern = r"- (.+?)\s\[(\d+(?:, \d+)*)\]"  # Matches the sentences and sources

    categories = re.findall(category_pattern, user_sheet)  # Extract headers
    category_dict = {category: [] for category in categories}  # Initialize dictionary for each category

    # Split the user_sheet into sections based on categories
    sections = re.split(category_pattern, user_sheet)

    # Iterate through sections and extract sentences and sources
    for i in range(1, len(sections), 2):  # Skip irrelevant parts
        category = sections[i].strip()  # Current category
        content = sections[i + 1]  # Content for the category
        matches = re.findall(content_pattern, content)  # Extract sentences and sources
        for match in matches:
            category_dict[category].append({
                "statement": match[0].strip(),
                "sources": literal_eval(f"[{match[1]}]")
            })

    # # Output the grouped dictionary
    # for category, items in category_dict.items():
    #     print(f"Category: {category}")
    #     for item in items:
    #         print(f"  Statement: {item['statement']}")
    #         print(f"  Sources: {item['sources']}")
    #     print()
    
    return category_dict

def get_source_wise_claims(category_dict, user_profile_list):
    '''
    organize claims based on sources using the user_profile_list
    '''
    source_wise_claims = defaultdict(list)

    for cat, claims in category_dict.items():
        for claim_info in claims: 
            # sources = claim_info['sources']
            # # get the minimum source number
            # min_source = min(sources)

            claim = claim_info['statement']
            # search for the claim in the user_profile_list
            for uctr, user_profile in enumerate(user_profile_list):
                if claim in user_profile:
                    break
            min_source = uctr + 1

            # update the sources with the minimum source number
            source_wise_claims[min_source].append((claim_info['statement'], cat))

            # # iterate over all sources
            # for source in sources:
            #     source_wise_claims[source].append((claim_info['statement'], cat))
    
    # sort source_wise_claims based on length of claims
    source_wise_claims = dict(sorted(source_wise_claims.items(), key=lambda x: len(x[1]), reverse=True))

    return source_wise_claims

def dump_annotation_sample(source_wise_claims, file):
    '''
    construct annotation data sample
    '''
    # consider top 3 sources
    top_sources = list(source_wise_claims.keys())[:3]

    # profile story 
    profile_story_path = f'../../datasets/data_splits/data/Reddit/profile/{file}'
    with open(profile_story_path, 'r') as f:
        profile_story = json.load(f)
    
    # output annotation dir 
    annotation_dir = 'annotation_data'
    if not os.path.exists(annotation_dir):
        os.makedirs(annotation_dir)
    
    # annotation file dir 
    annotation_file_dir = f'{annotation_dir}/{file.split('.')[0]}'
    if not os.path.exists(annotation_file_dir):
        os.makedirs(annotation_file_dir)
    
    rows = []

    # iterate over top sources
    for source in top_sources:
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
                "category": claim[1],
                "claim": claim[0],
                "coherence": '',
                "consistency": '',
                "comments": '',

            })
        
        # write source wp and story to a file
        source_info = f"#### Writing Prompt ####\n{source_wp}\n\n\n#### Story ####\n{source_story}"
        with open(f'{annotation_file_dir}/{source}.txt', 'w') as f:
            f.write(source_info)
    
    # write annotation data to a csv file
    df = pd.DataFrame(rows)
    df.to_csv(f'{annotation_file_dir}/annotation_sheet.csv', index=False)



def main():
    user_sheet_dir = '../../experiments/user_profile/delta_schema/Reddit/'

    organize_sheet_dir = 'clean_user_sheet'
    if not os.path.exists(organize_sheet_dir):
        os.makedirs(organize_sheet_dir)
    
    source_wise_claims_dir = 'source_wise_claims'
    if not os.path.exists(source_wise_claims_dir):
        os.makedirs(source_wise_claims_dir)

    # iterate over all users 
    for file in os.listdir(user_sheet_dir):
        if file.endswith('.json'):
            user_sheet_path = os.path.join(user_sheet_dir, file)
            with open(user_sheet_path, 'r') as f:
                user_profile_list = json.load(f)
            user_sheet = extract_writing_sheet(user_profile_list[-1], 'combined_user_sheet')
            category_dict = organize_user_sheet(user_sheet)
            with open(f'{organize_sheet_dir}/{file}', 'w') as f:
                json.dump(category_dict, f, indent=4)
            # source wise claims
            source_wise_claims = get_source_wise_claims(category_dict, user_profile_list)
            with open(f'{source_wise_claims_dir}/{file}', 'w') as f:
                json.dump(source_wise_claims, f, indent=4)
            # construct annotation data sample
            dump_annotation_sample(source_wise_claims, file)


if __name__ == '__main__':
    main()