'''
Implementation of methods for Story Generation
'''

import os 
from tqdm import tqdm
import json 
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
    
    def perform_vanilla(self, source='Reddit'):
        '''
        Perform Vanilla story generation
        '''
        def construct_vanilla_prompt(example):
            '''
            Construct the Vanilla prompt
            '''
            story_length = example['metadata']['story_length']
            # delete story length from metadata
            del example['metadata']['story_length']
            # writing prompt
            writing_prompt = example['writing_prompt']

            # construct the user instruction 
            user_instruction = f"Write a short story corresponding to the following writing prompt. The story should be {story_length} words long. Directly start with the story, do not say things like 'Here\'s' the story \n\n"
            if source == 'AO3':
                del example['metadata']['story_name']
                user_instruction += f"Here is the metadata (fandom, rating, warnings, and relationships) for the story: {example['metadata']}\n\n"
            # include source_constraints
            user_instruction += f"Here are some constrains that you must follow:\n{source_constraints}\n\n"
            # include writing prompt
            user_instruction += f"Writing Prompt: {writing_prompt}\n\nStory:\n"

            # construct OpenAI prompt
            prompt = construct_prompt_message(system_instructions, user_instruction)
            return prompt
    
        print('Method: Vanilla Story Generation')
        print(f'Source: {source}')

        # vanilla output directory
        vanilla_output_dir = f'{self.output_dir}/vanilla/{source}'
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
            # with open(profile_file_path, 'r') as f:
            #     profile_data = json.load(f)
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

                prompt = construct_vanilla_prompt(example)
                # call the OpenAI model
                response = prompt_openai(prompt, max_tokens=4096, temperature=0.7, top_p=0.95)
                results.append({'writing_prompt': example['writing_prompt'], 'story': response})
                
                # write the results to the output directory
                with open(output_file_path, 'w') as f:
                    json.dump(results, f, indent=4)


def main():
    # create an instance of the StoryGenMethods class
    story_gen_methods = StoryGenMethods()
    # perform Vanilla story generation
    story_gen_methods.perform_vanilla(source='Reddit')

if __name__ == '__main__':
    main()
            

            
