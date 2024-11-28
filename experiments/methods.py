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
import argparse
import re

class StoryGenMethods():
    def __init__(self):
        # initailize the directories
        
        # 1. data split directory
        self.data_split_dir = '../datasets/data_splits/data'

        # 2. Vanilla instructions directory
        self.vanilla_prompt_instructions_dir = 'instructions/vanilla'

        # 3. User Profile instructions directory
        self.user_profile_instructions_dir = 'instructions/construct_user_profile'

        # 4. Generate Story User Profile instructions directory
        self.generate_story_user_profile_instructions_dir = 'instructions/generate_story_user_profile'

        # output directory
        self.output_dir = 'results/'

        # user profile directory
        self.user_profile_dir = 'user_profile'
    
    def construct_user_instruction(self, example_raw, source='Reddit'):
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
        # if source_constraints:
        #     user_instruction += "The story must follow the above mentioned constraints (## Story Rules)."
        user_instruction += " Directly start with the story, do not say things like 'Here\'s' the story.\n\n"
        if source == 'AO3':
            del example['metadata']['story_name']
            user_instruction += f"Here is the metadata (fandom, rating, warnings, and relationships) for the story: {example['metadata']}\n\n"
        # include writing prompt
        user_instruction += f"Writing Prompt: {writing_prompt}\n\nStory:\n"

        return user_instruction


    def get_few_shot_examples(self, profile_data, example, source='Reddit', top_k=1):
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
            user_instruction = self.construct_user_instruction(profile_data[pindex], source)
            few_shot_examples[pctr] = {'User': user_instruction, 'Assistant': profile_data[pindex]['story']}
        
        return few_shot_examples

    def perform_story_generation(self, source='Reddit', few_shot=False, story_output_dir=None, source_constraints_dir = None, system_instructions=''):
        '''
        performs story generation given - source, few_shot, source_constraints, output_dir
        '''
        def construct_story_prompt(example, source_constraints, few_shot_examples=None):
            '''
            Construct the Vanilla prompt
            '''
            # include user constraints
            user_constraints = f"## Story Rules\n{source_constraints}\n\n"

            # construct the user instruction
            user_instruction = self.construct_user_instruction(example, source)


            # construct OpenAI prompt
            prompt = construct_prompt_message(system_instructions, user_instruction, user_constraints, few_shot_examples)
            return prompt
    

        if few_shot:
            suffix = '_few_shot'
        else:
            suffix = ''

        # profile directory 
        profile_dir = f'{self.data_split_dir}/{source}/profile'
        # test directory
        test_dir = f'{self.data_split_dir}/{source}/test' 
        # source constraints
        if '.txt' in source_constraints_dir:
            # read the user instructions
            with open(source_constraints_dir, 'r') as f:
                source_constraints = f.read()
            iterate_over_source_constraints = False
        else:
            iterate_over_source_constraints = True
        

        # NOTE: Edit the system instructions
        # 1. Mention source constraints
        # if source_constraints:
        system_instructions += ' Be sure to adhere to the ## Story Rules provided, as they define the specific elements of the writing style you are expected to mimic.'
        # 2. Mention few shot demonstrations (chat history)
        if few_shot:
            system_instructions += ' Also, follow the patterns and examples demonstrated in the provided few-shot chat history, as they illustrate the tone, style, and structure of the desired writing style of your story.'

        
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
            output_file_path = os.path.join(story_output_dir, file)

            # check if the output file already exists
            if os.path.exists(output_file_path):
                # read the output file
                with open(output_file_path, 'r') as f:
                    results = json.load(f)
            else:
                results = []
            
            # iterate over the test data
            for ictr, example in tqdm(enumerate(test_data), desc=f'Processing {file}', total=len(test_data)):
                
                # stop after 2 iterations
                if ictr > 1:
                    break

                # check if the example already exists in the results
                if ictr < len(results):
                    continue
            
                # few_shot 
                if few_shot:
                    few_shot_examples = self.get_few_shot_examples(profile_data, example, source=source, top_k=1)
                else:
                    few_shot_examples = None
                
                if iterate_over_source_constraints:
                    source_constraints_path = f'{source_constraints_dir}/{file.split(".")[0]}.txt'
                    # read the user instructions
                    with open(source_constraints_path, 'r') as f:
                        source_constraints_raw = f.read()
                        # TODO: Extract content between the tags <story_rules></story_rules>
                        source_constraints = re.search(r'<story_rules>(.*?)</story_rules>', source_constraints_raw, re.DOTALL).group(1)
                        # check if the source constraints are empty
                        if not source_constraints:
                            source_constraints = source_constraints_raw

                prompt = construct_story_prompt(example, source_constraints, few_shot_examples)
                # call the OpenAI model
                response = prompt_openai(prompt, max_tokens=4096, temperature=0.7, top_p=0.95)
                results.append({'writing_prompt': example['writing_prompt'], 'story': response})
                
                # write the results to the output directory
                with open(output_file_path, 'w') as f:
                    json.dump(results, f, indent=4)
    
    def perform_vanilla(self, source='Reddit', few_shot=False):
        '''
        Vanilla Story Generation
        '''

        if few_shot:
            suffix = '_few_shot'
        else:
            suffix = ''

        # output directory
        output_dir = f'{self.output_dir}/vanilla{suffix}/{source}'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # system instructions 
        system_instructions_path = f'{self.vanilla_prompt_instructions_dir}/system_prompts/{source}.txt'    
        # read the system instructions
        with open(system_instructions_path, 'r') as f:
            system_instructions = f.read()

        
        # source constraints path
        source_constraints_path = f'{self.vanilla_prompt_instructions_dir}/user_prompts/{source}.txt'
        
        print('Method: Vanilla Story Generation')
        print(f'Few Shot: {few_shot}')
        print(f'Source: {source}')

        # perform story generation
        self.perform_story_generation(source=source, few_shot=few_shot, story_output_dir=output_dir, source_constraints_dir=source_constraints_path, system_instructions=system_instructions)

    
    def no_schema_user_profile(self, source='Reddit'):
        '''
        User Profile (No Schema)
        '''

        def construct_user_profile_prompt(examples):
            '''
            Construct the Prompt for User Profile
            '''

            # construct the user instruction
            user_instruction = ''
            # iterate through each example
            for ectr, example in enumerate(examples): 
                # deep copy example['metadata']
                metadata = copy.deepcopy(example['metadata'])
                # delete story name from metadata
                if 'story_name' in metadata:
                    del metadata['story_name']
                # delete story length from metadata
                example_dict = {'metadata': metadata, 'writing_prompt': example['writing_prompt'], 'story': example['story']}
                user_instruction += f"Example {ectr + 1}:\n{json.dumps(example_dict, indent=4)}\n\n"

            # construct OpenAI prompt
            prompt = construct_prompt_message(system_instructions, user_instruction, user_constraints)
            return prompt

        
        print('Method: No Schema User Profile')
        print(f'Source: {source}')

        # user profile output directory
        user_profile_output_dir = f'{self.user_profile_dir}/no_schema/{source}'
        if not os.path.exists(user_profile_output_dir):
            os.makedirs(user_profile_output_dir)

        # profile directory
        profile_dir = f'{self.data_split_dir}/{source}/profile'
        # test directory
        test_dir = f'{self.data_split_dir}/{source}/test'

        # 1. Generate User Profile

        # sytem instructions
        system_instructions_path = f'{self.user_profile_instructions_dir}/system_prompts/no_schema.txt'
        # user instructions
        user_instructions_path = f'{self.user_profile_instructions_dir}/user_prompts/no_schema.txt'

        # read the system instructions
        with open(system_instructions_path, 'r') as f:
            system_instructions = f.read()
        
        # read the user instructions
        with open(user_instructions_path, 'r') as f:
            user_constraints = f.read()
        
        # iterate through each file in the profile directory
        for file in tqdm(os.listdir(profile_dir), desc='User Profile (No Schema)', total=len(os.listdir(profile_dir))):
            profile_file_path = os.path.join(profile_dir, file)
            # profile data
            with open(profile_file_path, 'r') as f:
                profile_data = json.load(f)
            
            # output file path
            output_file_path = os.path.join(user_profile_output_dir, file.split('.')[0] + '.txt')

            # check if the output file already exists
            if os.path.exists(output_file_path):
                # read the output file
                with open(output_file_path, 'r') as f:
                    user_profile_response = f.read()
            else:
                # select last three examples from the profile data
                examples = profile_data[-3:]
                # construct the prompt
                prompt = construct_user_profile_prompt(examples)
                # call the OpenAI model
                user_profile_response = prompt_openai(prompt, max_tokens=4096, temperature=0.0)

                # save the response to the output directory
                with open(output_file_path, 'w') as f:
                    f.write(user_profile_response)
        
        print('User Profile (No Schema) Generated')


        # TODO: Generate stories using the user profile generated above
        story_output_dir = f'{self.output_dir}/no_schema/{source}'
        if not os.path.exists(story_output_dir):
            os.makedirs(story_output_dir)
        
        # system instructions
        system_instructions_path = f'{self.generate_story_user_profile_instructions_dir}/system_prompts/{source}.txt'
        # read the system instructions
        with open(system_instructions_path, 'r') as f:
            system_instructions = f.read()


        print('Method: User Profile (No Schema) Story Generation')
        print(f'Few Shot: True')
        print(f'Source: {source}')

        self.perform_story_generation(source=source, few_shot=True, story_output_dir=story_output_dir, source_constraints_dir = user_profile_output_dir, system_instructions=system_instructions)




def parse_args():
    parser = argparse.ArgumentParser(description='Story Generation Methods')
    # few shot
    parser.add_argument('--few_shot', action='store_true', help='Few Shot Story Generation')
    # source
    parser.add_argument('--source', type=str, default='Reddit', help='Source of the data')
    # # user profile (no schema) 'store_true'
    # parser.add_argument('--no_schema', action='store_true', help='User Profile (No Schema)')
    # int choice
    parser.add_argument('--choice', type=int, default=1, help='Choice of the method: 1. Vanilla, 2. User Profile (No Schema)')
    return parser.parse_args()

def main():
    # parse arguments
    args = parse_args()
    # source
    source = args.source    
    # few shot 
    few_shot = args.few_shot
    # # method choice
    choice = args.choice
    # choice = 2
    # create an instance of the StoryGenMethods class
    story_gen_methods = StoryGenMethods()

    if choice == 1:
        # perform Vanilla story generation
        story_gen_methods.perform_vanilla(source=source, few_shot=few_shot)
    elif choice == 2:
        # User Profile (No Schema)
        story_gen_methods.no_schema_user_profile(source=source)

if __name__ == '__main__':
    main()
            

            
