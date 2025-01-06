'''
identify schema that are not valid
'''

import os 
import re
from ast import literal_eval
import json
import argparse

def extract_writing_sheet(sheet_output, key='combined_author_sheet'):
    '''
    extract text between the tags <user_writing_sheet></user_writing_sheet>
    '''            
    sheet = re.search(rf'<{key}>(.*?)</{key}>', sheet_output, re.DOTALL).group(1)
    if not sheet:
        sheet = sheet_output
    return sheet    

def decode_unicode_escapes(text):
    return text.encode('utf-8').decode('unicode_escape')

def organize_user_sheet_new(user_sheet):
    '''
    Category-wise organization of user sheet with statement and example combined.
    '''

    # Extract headers and content
    category_pattern = r"### \*\*(.+?)\*\*"  # Matches the category headers
    statement_pattern = r"\d+\. \*\*(.+?)\*\*"  # Matches the numbered statements
    example_pattern = r"- Evidence: (.+?) \[(\d+(?:, \d+)*)\]"  # Matches the examples and sources

    categories = re.findall(category_pattern, user_sheet)  # Extract headers
    category_dict = {category: [] for category in categories}  # Initialize dictionary for each category

    # Split the user_sheet into sections based on categories
    sections = re.split(category_pattern, user_sheet)

    # Iterate through sections and extract statements and examples
    for i in range(1, len(sections), 2):  # Skip irrelevant parts
        category = sections[i].strip()  # Current category
        content = sections[i + 1]  # Content for the category

        # Match statements and corresponding examples
        statements = re.findall(statement_pattern, content)
        examples = re.findall(example_pattern, content)

        for statement, (example, sources) in zip(statements, examples):
            category_dict[category].append({
                "statement": decode_unicode_escapes(statement.strip()),
                "example": decode_unicode_escapes(example.strip()),
                "sources": literal_eval(f"[{sources}]")
            })


    return category_dict

def analyze_sheet(category_wise_sheet):
    '''
    check if some example does not fit the format
    '''

    problem = False 

    # iterate over all categories 
    for category in category_wise_sheet: 
        for entry in category_wise_sheet[category]: 
            example = entry['example']
            if "in the story regarding" not in example.lower(): 
                problem = True 
                break 
    
    return problem

def main():
    sources = ['Reddit', 'AO3', 'Storium', 'narrativemagazine', 'newyorker'] 

    user_profile_root_dir = 'user_profile/delta_schema'

    problem_sheets = []

    # iterate over sources 
    for source in sources: 
        source_dir = f"{user_profile_root_dir}/{source}"
        # iterate over files in the directory 
        for file in os.listdir(source_dir): 
            # read the file 
            with open(f"{source_dir}/{file}", 'r') as f: 
                schema_list = json.load(f)
            try:
                if len(schema_list) == 1:
                    key = 'writing_style'
                else:
                    key = 'combined_author_sheet'
                writing_sheet = extract_writing_sheet(schema_list[-1], key)
                category_wise_sheet = organize_user_sheet_new(writing_sheet)
                if analyze_sheet(category_wise_sheet):
                    problem_sheets.append(f"{source}/{file}")
            except AttributeError:
                print(f"AttributeError: {source}/{file}")
    
    print(problem_sheets)
    print(len(problem_sheets))

if __name__ == '__main__':
    main()