'''
examples for the user_sheet_annotation module
'''

import os
import pandas as pd
import re
import json 
import random
from ast import literal_eval
from collections import defaultdict
import argparse
import unicodedata


def extract_writing_sheet(sheet_output, key='combined_author_sheet'):
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
    example_pattern = r"- Evidence: (.+?) \[(\d+(?:, \d+)*)\]"  # Matches the examples and sources

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

def get_story_wise_claims(category_dict):
    '''
    organize claims based on sources
    '''
    story_wise_claims = defaultdict(list)

    for cat, claims in category_dict.items():
        for claim_info in claims: 
            # sources = claim_info['sources']
            # # get the minimum source number
            # min_source = min(sources)

            claim = claim_info['statement']
            example = claim_info['example']
            source = int(claim_info['sources'][0])

            # append to the story_wise_claims
            story_wise_claims[source].append((claim_info['statement'], claim_info['example'], cat))
    
    # sort story_wise_claims based on length of claims
    story_wise_claims = dict(sorted(story_wise_claims.items(), key=lambda x: len(x[1]), reverse=True))

    return story_wise_claims

def decode_unicode_escapes(text):
    return text.encode('utf-8').decode('unicode_escape')

def clean_text(text):
    """
    Clean up the text to make it readable by:
    1. Replacing Unicode characters with ASCII equivalents.
    2. Removing stray or problematic characters.
    """
    # Normalize text to decompose Unicode characters (e.g., é -> e)
    normalized_text = unicodedata.normalize("NFKD", text)
    
    # Encode to ASCII, ignoring characters that can't be converted
    ascii_text = normalized_text.encode("ascii", "ignore").decode("ascii")
    
    return ascii_text


def dump_annotation_sample(story_wise_claims, file, source='Reddit', claim_threshold=3, story_threshold=3, labelstudio_rows=None):
    '''
    construct annotation data sample
    '''
    # consider top 3 storys
    top_stories = list(story_wise_claims.keys())[:story_threshold]

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
    user_rows = []

    # iterate over top storys
    for story in top_stories:
        # number of claims should be atleast claim_threshold
        if len(story_wise_claims[story]) < claim_threshold:
            continue

        # get story wp and story 
        story_wp = profile_story[story-1]["writing_prompt"]
        # story_text = (
        #     profile_story[story-1]["story"]
        #     .encode("latin1", errors="ignore")  # Treat as Latin-1, ignoring invalid bytes
        #     .decode("utf-8", errors="ignore")  # Decode to UTF-8, ignoring undecodable bytes
        # )

        # story_text = clean_text(profile_story[story-1]["story"])

        story_text = profile_story[story-1]["story"]

        # Normalize carriage returns and literal `\n` first
        story_text = story_text.replace('\r', '')  # Remove carriage returns
        story_text = re.sub(r'[ \t]*\n[ \t]*', '\n', story_text)  # Normalize spaces around newlines
        story_text = story_text.replace('\\n', '\n')  # Replace escaped \n with actual newlines
        # Clean the story_text by replacing multiple newlines with a single newline
        story_text = re.sub(r'\n+', '\n', story_text).strip()
        # remove escape characters 
        story_text = story_text.replace('\\\"', '\"')
        story_text = story_text.replace('\\\\', '\\')
        
        # write story wp and story to a file
        story_info = f"#### Writing Prompt ####\n{story_wp}\n\n\n#### Story ####\n{story_text}"
        with open(f'{annotation_file_dir}/{story}.txt', 'w') as f:
            f.write(story_info)

        claims = story_wise_claims[story]
        # iterate over claims
        for cctr, claim in enumerate(claims):
            rows.append({
                "story_id": story,
                "category": claim[2],
                "claim": claim[0],
                "evidence": claim[1],
                "infer": '',
                "support": '',
                "comments": '',

            })

            # TODO: write to labelstudio format
            if labelstudio_rows is not None:
                append_dict = {
                    "user": f"{source}_{file}",
                    "story_id": story,
                    "claim_id": cctr,
                    "writing_prompt": story_wp,
                    "story": story_text,
                    "claim": claim[0],
                    "evidence": claim[1]
                }
                # add to labelstudio_rows
                labelstudio_rows.append(append_dict)
                # add to user_rows
                user_rows.append(append_dict)

    
    # write annotation data to a csv file
    df = pd.DataFrame(rows)
    df.to_csv(f'{annotation_file_dir}/annotation_sheet.csv', index=False)

    # return user_rows
    return user_rows

def calculate_stats(labelstudio_rows):
    '''
    calculate statistics for labelstudio format
    '''
    # unique prompts
    prompts = set()
    # unique stories
    stories = set()
    # unique users 
    users = set()
    for row in labelstudio_rows:
        prompts.add(row['writing_prompt'])
        stories.add(row['story'])
        users.add(row['user'])
    
    print('\n\n### LabelStudio Statistics ###\n')
    print('Unique users:', len(users))
    print('Unique prompts:', len(prompts))
    print('Unique stories:', len(stories))
    print('Total claims:', len(labelstudio_rows))


def prepare_annotator_data(source_user_dict, num_annotators, max_claims, story_per_annotator):
    '''
    select data for annotators (round robin fashion)
    '''

    annotator_data = defaultdict(dict)

    # NOTE: choose 12 stories common for every annotator 
    ctr = 0
    # iterate over sources
    break_flag = False
    while True:
        for source in source_user_dict:
            # choose a random user
            user = random.choice(list(source_user_dict[source].keys()))
            # choose stories (randomly max_claims)
            if len(source_user_dict[source][user]) < max_claims:
                # shuffle the list
                choose_stories = random.sample(source_user_dict[source][user], len(source_user_dict[source][user]))
            else:
                choose_stories = random.sample(source_user_dict[source][user], max_claims)

            for i in range(num_annotators):
                # append to annotator_i
                annotator_data[i][f"{source}_{user}"] = choose_stories

            # remove the selected user
            del source_user_dict[source][user]

            # increment counter
            ctr += 1

            # breaking condition
            if ctr >= 12:
                break_flag = True
                break
        
        if break_flag:
            break
        
    # NOTE: choose 10 more stories for every annotator
    annotator_flags = defaultdict(int)
    while True:
        # iterate over sources
        for source in source_user_dict:
            # iterate over annotators
            for i in range(num_annotators):
                # check if no more users exist
                if len(source_user_dict[source]) == 0:
                    break
                # check if annotator data exceeds threshold
                if len(annotator_data[i]) >= story_per_annotator:
                    annotator_flags[i] = 1
                    continue
                # choose a random user
                user = random.choice(list(source_user_dict[source].keys()))
                # choose stories (randomly max_claims)
                if len(source_user_dict[source][user]) < max_claims:
                    # shuffle the list
                    choose_stories = random.sample(source_user_dict[source][user], len(source_user_dict[source][user]))
                else:
                    choose_stories = random.sample(source_user_dict[source][user], max_claims)
                
                # append to annotator_i
                annotator_data[i][f"{source}_{user}"] = choose_stories

                # remove the selected user
                del source_user_dict[source][user]

        # breaking condition 
        # iterate over the annotators
        all_flags = True 
        for i in range(num_annotators):
            if annotator_flags[i] == 0:
                all_flags = False
    
        if all_flags:
            break


    return annotator_data

def get_annotator_data_stats(annotator_data, num_annotators):
    '''
    get annotator data statistics
    '''
    # iterate over annotators

    print('\n\n### Annotator Data Statistics ###\n')

    unique_users = set()
    unique_claims = list()
    common_users = set()
    common_claims = []
    for i in range(num_annotators):
        print(f'\n### Annotator {i} ###\n')
        print('Total users:', len(annotator_data[i]))
        num_claims = []
        for user, claims in annotator_data[i].items():
            num_claims.append(len(claims))
            # check if user in unique_users
            if user in unique_users and user not in common_users:
                common_claims.extend(claims)
                common_users.add(user)
            
            # add unique claims
            unique_claims.extend([claim['evidence'] for claim in claims])
        
            unique_users.add(user)

        
        # average number of claims per user
        print('Total claims:', sum(num_claims))
        print('Average number of claims per user:', sum(num_claims)/len(num_claims))
    
    print('\n\n### Common Users ###\n')
    print('Total common users:', len(common_users))
    print('Total common claims:', len(common_claims))

    print('\n\n### Unique Users ###\n')
    print('Total unique users:', len(unique_users))
    print('Total unique claims:', len(set(unique_claims)))

    # if not os.path.exists('upwork_annotator_data'):
    #     os.makedirs('upwork_annotator_data')

    # with open('upwork_annotator_data/common_users.json', 'w') as f:
    #     json.dump(list(common_users), f, indent=4)
    

def parse_args():
    '''
    Parse command line arguments
    '''
    parser = argparse.ArgumentParser(description='Extract examples for annotation')
    # parser.add_argument('--source', type=str, default='Reddit', help='Source: Reddit, AO3, Storium, narrativemagazine, newyorker')
    # claim_threshold (int)
    parser.add_argument('--claim_threshold', type=int, default=3, help='claim_threshold for minimum number of claims per story')
    parser.add_argument('--story_threshold', type=int, default=1, help='story_threshold for maximum number of stories per user')
    # number of annotators 
    parser.add_argument('--num_annotators', type=int, default=3, help='number of annotators')
    # max number of claims per story
    parser.add_argument('--max_claims', type=int, default=6, help='max number of claims per story')
    # story per annotator 
    parser.add_argument('--story_per_annotator', type=int, default=22, help='number of stories per annotator')
    # store true if you want to dump labelstudio format
    parser.add_argument('--labelstudio', action='store_true', help='store true if you want to dump labelstudio format')
    args = parser.parse_args()
    return args

def main():
    # parse arguments
    args = parse_args()

    claim_threshold = args.claim_threshold
    story_threshold = args.story_threshold
    labelstudio = args.labelstudio
    num_annotators = args.num_annotators
    max_claims = args.max_claims
    story_per_annotator = args.story_per_annotator

    # set random seed
    random.seed(37)

    if labelstudio:
        labelstudio_rows = []
    else:
        labelstudio_rows = None

    print('claim_threshold:', claim_threshold)
    print('story_threshold:', story_threshold)

    sources = ['Reddit', 'AO3', 'Storium', 'narrativemagazine', 'newyorker']
    # store data for annotation
    source_user_dict = dict()
    for source in sources:
        source_user_dict[source] = defaultdict(list)

    # iterate over all sources
    for source in sources:
        print(f'Processing {source}...')
        user_sheet_dir = f'../../experiments/user_profile/delta_schema/{source}/'

        organize_sheet_dir = f'clean_user_sheet/{source}'
        if not os.path.exists(organize_sheet_dir):
            os.makedirs(organize_sheet_dir)
        
        # story_wise_claims_dir = f'story_wise_claims/{source}'
        # if not os.path.exists(story_wise_claims_dir):
        #     os.makedirs(story_wise_claims_dir)

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
                    key = 'combined_author_sheet'
                user_sheet = extract_writing_sheet(user_profile_list[-1], key)
                category_dict = organize_user_sheet(user_sheet)
                with open(f'{organize_sheet_dir}/{file}', 'w') as f:
                    json.dump(category_dict, f, indent=4)
                # source wise claims
                story_wise_claims = get_story_wise_claims(category_dict)
                # with open(f'{story_wise_claims_dir}/{file}', 'w') as f:
                #     json.dump(story_wise_claims, f, indent=4)
                # construct annotation data sample
                user_rows = dump_annotation_sample(story_wise_claims, file, source, claim_threshold, story_threshold, labelstudio_rows)
                # add to source_user_dict
                if len(user_rows) > 0:
                    source_user_dict[source][file] = user_rows
    
    
    # calculate statistics for labelstudio format
    calculate_stats(labelstudio_rows)

    # dump source_user_dict to a file
    with open('annotation_data/source_user_dict.json', 'w') as f:
        json.dump(source_user_dict, f, indent=4)
    
    if labelstudio:
        # write to labelstudio format
        df = pd.DataFrame(labelstudio_rows)
        df.to_csv('annotation_data/labelstudio_format.csv', index=False, encoding='utf-8')

        # save as JSON
        with open('annotation_data/labelstudio_format.json', 'w', encoding='utf-8') as f:
            json.dump(labelstudio_rows, f, ensure_ascii=False, indent=4)

    # prepare annotator data
    annotator_data = prepare_annotator_data(source_user_dict, num_annotators, max_claims, story_per_annotator)

    # get annotator data statistics
    get_annotator_data_stats(annotator_data, num_annotators)

    # annotator data directory 
    annotator_data_dir = 'upwork_annotator_data'
    if not os.path.exists(annotator_data_dir):
        os.makedirs(annotator_data_dir)
    
    # iterate over annotators
    for i in range(num_annotators):
        # write to a file
        with open(f'{annotator_data_dir}/annotator_{i}.json', 'w') as f:
            annotation_data = []
            for user, data in annotator_data[i].items():
                annotation_data.extend(data)
            json.dump(annotation_data, f, indent=4)


if __name__ == '__main__':
    main()