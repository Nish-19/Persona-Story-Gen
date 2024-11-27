'''
Implementation of methods for Story Generation
'''

import os 
from tqdm import tqdm
import json 
import copy
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from prompt_llm_utils import construct_prompt_message, prompt_openai

class StoryGenMethods():
    def __init__(self):
        # initailize the directories
        
        # 1. data split directory
        self.data_split_dir = '../datasets/data_splits/data'

        # 2. Vanilla directory
        self.vanilla_prompt_dir = 'instructions/vanilla'

        # output directory
        self.output_dir = 'results/'
    
    def construct_user_instruction(self, example_raw, source_constraints=None, source='Reddit'):
        '''
        Construct the user instruction
        '''
        # copy example_raw to example to avoid modifying the original example 
        example = copy.deepcopy(example_raw)
        story_length = example['metadata']['story_length']
        # delete story length from metadata
        del example['metadata']['story_length']
        # writing prompt
        writing_prompt = example['writing_prompt']
        # construct the user instruction 
        user_instruction = f"Write a short story corresponding to the following writing prompt. The story should be {story_length} words long."
        if source_constraints:
            user_instruction += "The story must follow the above mentioned constraints (## Story Constraints)."
        user_instruction += "Directly start with the story, do not say things like 'Here\'s' the story \n\n"
        if source == 'AO3':
            del example['metadata']['story_name']
            user_instruction += f"Here is the metadata (fandom, rating, warnings, and relationships) for the story: {example['metadata']}\n\n"
        # include writing prompt
        user_instruction += f"Writing Prompt: {writing_prompt}\n\nStory:\n"

        return user_instruction


    def get_few_shot_examples(self, profile_data, example, source_constraints=None, source='Reddit', top_k=1):
        '''
        return the few shot examples
        '''
        # get most similar examples from the profile data using BM25
        profile_prompts = [p['writing_prompt'] for p in profile_data]
        query = example['writing_prompt']

        # Tokenize the prompts and query
        stop_words = set(stopwords.words('english'))
        tokenized_prompts = [[word for word in word_tokenize(prompt.lower()) if word not in stop_words] for prompt in profile_prompts]
        tokenized_query = [word for word in word_tokenize(query.lower()) if word not in stop_words]

        # Perform BM25
        bm25 = BM25Okapi(tokenized_prompts)
        scores = bm25.get_scores(tokenized_query)
        profile_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # construct few shot examples
        few_shot_examples = {}
        for pctr, pindex in enumerate(profile_indices):
            # construct user instruction 
            user_instruction = self.construct_user_instruction(profile_data[pindex], source_constraints, source)
            few_shot_examples[pctr] = {'User': user_instruction, 'Assistant': profile_data[pindex]['story']}
        
        return few_shot_examples
    
    def perform_vanilla(self, source='Reddit', few_shot=False):
        '''
        Perform Vanilla story generation
        '''
        def construct_vanilla_prompt(example, few_shot_examples=None):
            '''
            Construct the Vanilla prompt
            '''
            # include user constraints
            user_constraints = f"## Story Constraints\n{source_constraints}\n\n"

            # construct the user instruction
            user_instruction = self.construct_user_instruction(example, source_constraints, source)

            # construct OpenAI prompt
            prompt = construct_prompt_message(system_instructions, user_instruction, user_constraints, few_shot_examples)
            return prompt
    
        print('Method: Vanilla Story Generation')
        print(f'Few Shot: {few_shot}')
        print(f'Source: {source}')

        if few_shot:
            suffix = '_few_shot'
        else:
            suffix = ''

        # vanilla output directory
        vanilla_output_dir = f'{self.output_dir}/vanilla{suffix}/{source}'
        if not os.path.exists(vanilla_output_dir):
            os.makedirs(vanilla_output_dir)

        # profile directory 
        profile_dir = f'{self.data_split_dir}/{source}/profile'
        # test directory
        test_dir = f'{self.data_split_dir}/{source}/test' 
        # system instructions 
        system_instructions_path = f'{self.vanilla_prompt_dir}/system_prompts/{source}.txt'    
        # source constraints
        source_constraints_path = f'{self.vanilla_prompt_dir}/user_prompts/{source}.txt'

        # read the system instructions
        with open(system_instructions_path, 'r') as f:
            system_instructions = f.read()
        
        # read the user instructions
        with open(source_constraints_path, 'r') as f:
            source_constraints = f.read()
        
        # iterate through each file in the profile directory
        for ctr, file in tqdm(enumerate(os.listdir(test_dir)), total=len(os.listdir(test_dir)), desc='Vanilla Story Generation'):

            # # break after 1 iteration
            # if ctr > 0:
            #     break

            profile_file_path = os.path.join(profile_dir, file)
            test_file_path = os.path.join(test_dir, file)

            # profile data
            with open(profile_file_path, 'r') as f:
                profile_data = json.load(f)
            
            # test data
            with open(test_file_path, 'r') as f:
                test_data = json.load(f)
            
            # output file path
            output_file_path = os.path.join(vanilla_output_dir, file)

            # check if the output file already exists
            if os.path.exists(output_file_path):
                # read the output file
                with open(output_file_path, 'r') as f:
                    results = json.load(f)
            else:
                results = []
            
            # iterate over the test data
            for ictr, example in tqdm(enumerate(test_data), desc=f'Processing {file}', total=len(test_data)):
                # check if the example already exists in the results
                if ictr < len(results):
                    continue
            
                # few_shot 
                if few_shot:
                    few_shot_examples = self.get_few_shot_examples(profile_data, example, source_constraints=source_constraints, source=source, top_k=1)
                else:
                    few_shot_examples = None

                prompt = construct_vanilla_prompt(example, few_shot_examples)
                # call the OpenAI model
                response = prompt_openai(prompt, max_tokens=4096, temperature=0.7, top_p=0.95)
                results.append({'writing_prompt': example['writing_prompt'], 'story': response})
                
                # write the results to the output directory
                with open(output_file_path, 'w') as f:
                    json.dump(results, f, indent=4)


def main():
    # few shot 
    few_shot = True
    # create an instance of the StoryGenMethods class
    story_gen_methods = StoryGenMethods()
    # perform Vanilla story generation
    story_gen_methods.perform_vanilla(source='Reddit', few_shot=few_shot)

if __name__ == '__main__':
    main()
            

            
