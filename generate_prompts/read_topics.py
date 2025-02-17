'''
analyze topics
'''

import os
import json 
import matplotlib.pyplot as plt
from collections import Counter
from ast import literal_eval

def main():
    sources = ['AO3', 'narrativemagazine', 'newyorker', 'Reddit', 'Storium']
    source_alias = {'AO3': 'AO3', 'narrativemagazine': 'N.Magazine', 'newyorker': 'New Yorker', 'Reddit': 'Reddit', 'Storium': 'Storium'}

    output_dir = 'read_topics'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plot_dir = f"{output_dir}/wp_themes"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

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
        total_count = sum(topic_counter.values())

        # sort the topics based on the count descending
        sorted_topics = {topic: count for topic, count in sorted(topic_counter.items(), key=lambda x: x[1], reverse=True)}

        # write the topics to a json file
        out_file = f'{output_dir}/{source}.json'
        with open(out_file, 'w') as f:
            json.dump(sorted_topics, f, indent=4)
        
        # Construct plot with distribution of top 10 topics (normalized)
        top_10_topics = dict(list(sorted_topics.items())[:10])
        top_10_normalized = {topic: count / total_count for topic, count in top_10_topics.items()}

        # Plot bar chart with narrower bars
        plt.figure(figsize=(6, 4))  # Reduce figure width
        top_10_values_per = [value * 100 for value in top_10_normalized.values()]
        plt.bar(top_10_normalized.keys(), top_10_values_per, width=0.5)  # Reduce bar width
        plt.xlabel("Themes", fontsize=12)
        plt.ylabel("Normalized Frequency (%)", fontsize=12)
        plt.title(f"{source_alias[source]}", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=14)
        plt.tight_layout()

        # Save the plot
        plot_file = f"{plot_dir}/{source}.png"
        plt.savefig(plot_file, dpi=300)  # Increase resolution for better readability
        plt.close()

if __name__ == '__main__':
    main()
