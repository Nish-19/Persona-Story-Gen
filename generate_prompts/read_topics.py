'''
analyze topics
'''

import os
import json 
from collections import Counter
from ast import literal_eval

def main():
    sources = ['AO3', 'narrativemagazine', 'newyorker', 'Reddit', 'Storium']

    output_dir = 'read_topics'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # iterate over sources
    for source in sources:
        inp_file = f'analyzed_prompts/{source}.json'

        # read file
        with open(inp_file, 'r') as f:
            data = json.load(f) 
        
        all_topics = []
        # iterate over the data
        for key, topics_str in data.items():
            topics = literal_eval(topics_str.strip('`'))
            all_topics.extend(topics)
        
        # count the topics
        topic_counter = Counter(all_topics)
        # sort the topics based on the count descending
        sorted_topics = {topic: count for topic, count in sorted(topic_counter.items(), key=lambda x: x[1], reverse=True)}

        # write the topics to a json file
        out_file = f'{output_dir}/{source}.json'
        with open(out_file, 'w') as f:
            json.dump(sorted_topics, f, indent=4)

        

if __name__ == '__main__':
    main()