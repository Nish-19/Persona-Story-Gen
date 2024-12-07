'''
examples for the user_sheet_annotation module
'''

import os
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

def get_source_wise_claims(category_dict):
    '''
    organize claims based on sources
    '''
    source_wise_claims = defaultdict(list)

    for cat, claims in category_dict.items():
        for claim_info in claims: 
            sources = claim_info['sources']
            # get the minimum source number
            min_source = min(sources)
            # update the sources with the minimum source number
            source_wise_claims[min_source].append((claim_info['statement'], cat))

            # # iterate over all sources
            # for source in sources:
            #     source_wise_claims[source].append((claim_info['statement'], cat))
    
    # sort source_wise_claims based on length of claims
    source_wise_claims = dict(sorted(source_wise_claims.items(), key=lambda x: len(x[1]), reverse=True))

    return source_wise_claims


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
            source_wise_claims = get_source_wise_claims(category_dict)
            with open(f'{source_wise_claims_dir}/{file}', 'w') as f:
                json.dump(source_wise_claims, f, indent=4)


if __name__ == '__main__':
    main()