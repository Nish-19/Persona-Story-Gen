'''
analyze dataset
'''

import os 
import pandas as pd
import json 
from collections import Counter, defaultdict

def main():
    reddit_dir = 'Reddit/selected_human' 
    unique_prompts = set()
    unique_prompt_per_user = defaultdict(list)
    user_wise_length = defaultdict(list)
    user_wise_avg = defaultdict(float)
    # iterate over all files in the directory
    for file in os.listdir(reddit_dir):
        file_path = f"{reddit_dir}/{file}"
        user_name = file.split('.')[0]
        # read file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # iterate over all posts
        for post in data:
            comment_text = post['comment']
            prompt = post['post_title']
            # length of the comment
            comment_length = len(comment_text.split())
            user_wise_length[user_name].append(comment_length)
            # add prompt to the set
            unique_prompts.add(prompt)
            # check if current prompt already exists in the list
            if prompt not in unique_prompt_per_user[user_name]:
                unique_prompt_per_user[user_name].append(prompt)
            else:
                print(f'Duplicate prompt found: {file}: {prompt}')
        
        # calculate average length of comments for each user
        user_wise_avg[user_name] = sum(user_wise_length[user_name])/len(user_wise_length[user_name])
    
    total_avg = sum([avg for avg in user_wise_avg.values()])/len(user_wise_avg)



    # Assuming unique_prompts, user_wise_avg, and total_avg are already defined
    data = [{'Metric': 'Total Number of unique prompts', 'Average Length': len(unique_prompts)}]

    for user, avg in user_wise_avg.items():
        data.append({'Metric': user, 'Average Length': round(avg, 2)})

    data.append({'Metric': 'All users', 'Average Length': round(total_avg, 2)})

    df = pd.DataFrame(data)
    df.to_csv('Reddit/analysis.csv', index=False)

if __name__ == '__main__':
    main()