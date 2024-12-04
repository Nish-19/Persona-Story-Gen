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

        # 4. Rule extractor instructions directory
        self.rule_extractor_instructions_dir = 'instructions/rule_extractor'

        # 4. Generate Story User Profile instructions directory
        self.generate_story_user_profile_instructions_dir = 'instructions/generate_story_user_profile'

        # output directory
        self.output_dir = 'results/'

        # output directory (profile)
        self.output_dir_profile = 'results_profile/'

        # user profile directory
        self.user_profile_dir = 'user_profile'

        # user sheets directory
        self.user_sheets_dir = 'user_sheets'

        # story rules directory
        self.story_rules_dir = 'story_rules'
    
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
        
        return few_shot_examples, profile_indices

    def perform_story_generation(self, source='Reddit', few_shot=False, story_output_dir=None, source_constraints_dir = None, system_instructions='', debug=False, is_profile=False, few_shot_top_k=1):
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
            prompt = construct_prompt_message(system_instructions, user_instruction, user_constraints, few_shot_examples, add_at_end=True)
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
        for fctr, file in tqdm(enumerate(os.listdir(test_dir)), total=len(os.listdir(test_dir)), desc='Story Generation'):

            # break after 3 iterations
            if debug:
                if fctr > 2:
                    break

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

            if is_profile:
                consider_data = profile_data
            else:
                consider_data = test_data
            
            # iterate over the test data
            for ictr, example in tqdm(enumerate(consider_data), desc=f'Processing {file}', total=len(consider_data)):
                
                # stop after 2 iterations
                if debug:
                    if ictr > 2:
                        break

                # check if the example already exists in the results
                if ictr < len(results):
                    continue
            
                if len(profile_data) == 0:
                    continue
            
                # few_shot 
                if few_shot:
                    few_shot_examples, _ = self.get_few_shot_examples(profile_data, example, source=source, top_k=few_shot_top_k)
                else:
                    few_shot_examples = None

                # Get source constraints for this user
                if iterate_over_source_constraints:
                    source_constraints_path = f'{source_constraints_dir}/{file.split(".")[0]}.txt'
                    # check if the source constraints file exists
                    if os.path.exists(source_constraints_path):
                        # read the user instructions
                        with open(source_constraints_path, 'r') as f:
                            source_constraints_raw = f.read()
                    else:
                        source_constraints_path = f'{source_constraints_dir}/{file}'
                        # read the user instructions
                        try:
                            with open(source_constraints_path, 'r') as f:
                                all_source_constraints = json.load(f)
                            
                            source_constraints_raw = all_source_constraints[ictr]
                            # TODO: Extract content between the tags <story_rules></story_rules>
                            source_constraints = re.search(r'<story_rules>(.*?)</story_rules>', source_constraints_raw, re.DOTALL).group(1)
                            # check if the source constraints are empty
                            if not source_constraints:
                                source_constraints = source_constraints_raw
                        except Exception as e:
                            continue
                        

                prompt = construct_story_prompt(example, source_constraints, few_shot_examples)
                # call the OpenAI model
                try:
                    response = prompt_openai(prompt, max_tokens=4096, temperature=0.7, top_p=0.95)
                except Exception as e:
                    response = None
                results.append({'writing_prompt': example['writing_prompt'], 'story': response})
                
                # write the results to the output directory
                with open(output_file_path, 'w') as f:
                    json.dump(results, f, indent=4)
    
    def perform_vanilla(self, source='Reddit', few_shot=False, debug=False, is_profile=False):
        '''
        Vanilla Story Generation
        '''

        if few_shot:
            suffix = '_few_shot'
        else:
            suffix = ''

        # output directory
        if not is_profile:
            output_dir = f'{self.output_dir}/vanilla{suffix}/{source}'
        else:
            output_dir = f'{self.output_dir_profile}/vanilla{suffix}/{source}'

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
        if is_profile:
            print('Profile Data')
        print(f'Few Shot: {few_shot}')
        print(f'Source: {source}')

        # perform story generation
        self.perform_story_generation(source=source, few_shot=few_shot, story_output_dir=output_dir, source_constraints_dir=source_constraints_path, system_instructions=system_instructions, debug=debug, is_profile=is_profile)

    def perform_oracle_rules(self, source='Reddit', debug=False):
        '''
        Oracle Rules Story Generation
        1. generate oracle delta rules for the test set 
        2. use these rules to generate stories
        '''
        pass 

    
    def no_schema_user_profile(self, source='Reddit', debug=False):
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

        self.perform_story_generation(source=source, few_shot=True, story_output_dir=story_output_dir, source_constraints_dir = user_profile_output_dir, system_instructions=system_instructions, debug=debug)
    
    def schema_user_profile(self, source='Reddit', debug=False, few_shot_top_k=1):
        '''
        User Profile (Schema)
        '''

        def construct_user_sheet_prompt(example):
            '''
            Construct the Prompt for User Profile
            '''

            # construct the user instruction
            user_instruction = ''
            # deep copy example['metadata']
            metadata = copy.deepcopy(example['metadata'])
            # delete story name from metadata
            if 'story_name' in metadata:
                del metadata['story_name']
            # delete story length from metadata
            example_dict = {'metadata': metadata, 'writing_prompt': example['writing_prompt'], 'story': example['story']}
            user_instruction += f"{json.dumps(example_dict, indent=4)}\n\n"

            # construct OpenAI prompt
            prompt = construct_prompt_message(system_instructions_sheet, user_instruction, user_constraints_sheet)
            return prompt

        def construct_user_profile_prompt(prev_prompt_dict, current_prompt_dict):
            '''
            Construct the Prompt for User Profile
            '''

            # construct the user instruction
            user_instruction = ''
            input_prompt_dict = {'Previous_Information': prev_prompt_dict, 'Current_Information': current_prompt_dict}
            user_instruction += f"{json.dumps(input_prompt_dict, indent=4)}\n\n"

            # construct OpenAI prompt
            prompt = construct_prompt_message(system_instructions_combine, user_instruction, user_constraints_combine)
            return prompt
        

        def construct_story_rules_prompt(writing_prompt, user_profile): 
            '''
            Construct the Prompt for Story Rules
            '''
            # construct the user instruction
            user_instruction_dict = {'writing_prompt': writing_prompt, 'user_profile': user_profile}
            user_instruction = f"{json.dumps(user_instruction_dict, indent=4)}\n\n"

            # construct OpenAI prompt
            prompt = construct_prompt_message(system_instructions_story_rules, user_instruction, user_constraints_story_rules)
            return prompt
    
        def extract_writing_sheet(sheet_output, key='combined_user_sheet'):
            '''
            extract text between the tags <user_writing_sheet></user_writing_sheet>
            '''            
            sheet = re.search(rf'<{key}>(.*?)</{key}>', sheet_output, re.DOTALL).group(1)
            if not sheet:
                sheet = sheet_output
            return sheet
        
    
        print('Method: Schema User Profile')
        print(f'Source: {source}')

        # user sheets output directory
        user_sheets_output_dir = f'{self.user_sheets_dir}/{source}'
        if not os.path.exists(user_sheets_output_dir):
            os.makedirs(user_sheets_output_dir)
        
        # user profile output directory
        user_profile_output_dir = f'{self.user_profile_dir}/schema/{source}'
        if not os.path.exists(user_profile_output_dir):
            os.makedirs(user_profile_output_dir)
        
        # profile directory
        profile_dir = f'{self.data_split_dir}/{source}/profile'
        # test directory
        test_dir = f'{self.data_split_dir}/{source}/test'

        # NOTE: STEP 1: Generate the User Writing Sheets
        # sytem instructions
        system_instructions_sheet_path = f'{self.user_profile_instructions_dir}/system_prompts/schema.txt'
        # user instructions
        user_instructions_sheet_path = f'{self.user_profile_instructions_dir}/user_prompts/schema.txt'

        # read the system instructions
        with open(system_instructions_sheet_path, 'r') as f:
            system_instructions_sheet = f.read()
        
        # read the user instructions
        with open(user_instructions_sheet_path, 'r') as f:
            user_constraints_sheet = f.read()
        
        # iterate through each file in the profile directory
        for file in tqdm(os.listdir(profile_dir), desc='User Sheet (Schema)', total=len(os.listdir(profile_dir))):
            profile_file_path = os.path.join(profile_dir, file)
            # profile data
            with open(profile_file_path, 'r') as f:
                profile_data = json.load(f)
            
            # output file path
            output_file_path = os.path.join(user_sheets_output_dir, file)

            # check if the output file already exists
            if os.path.exists(output_file_path):
                # read the output file
                with open(output_file_path, 'r') as f:
                    user_sheet_response = json.load(f)
            else:
                user_sheet_response = []
            
            # iterate through each example in the profile data
            for ectr, example in tqdm(enumerate(profile_data), desc=f'Processing {file}', total=len(profile_data)):
                # check if the example already exists in the user sheet response
                if ectr < len(user_sheet_response):
                    continue

                # # break after 3 iterations
                # if ectr > 2:
                #     break
                
                # construct the prompt
                prompt = construct_user_sheet_prompt(example)
                # call the OpenAI model
                try:
                    response = prompt_openai(prompt, max_tokens=4096, temperature=0.0)
                except Exception as e:
                    response = None
                user_sheet_response.append(response)

                # write the results to the output directory
                with open(output_file_path, 'w') as f:
                    json.dump(user_sheet_response, f, indent=4)
        
        # NOTE: STEP 2: Generate User Profiles using the User Writing Sheets

        # system instructions
        system_instructions_combine_path = f'{self.user_profile_instructions_dir}/system_prompts/combine.txt'
        # user instructions
        user_instructions_combine_path = f'{self.user_profile_instructions_dir}/user_prompts/combine.txt'

        # read the system instructions
        with open(system_instructions_combine_path, 'r') as f:
            system_instructions_combine = f.read()
        
        # read the user instructions
        with open(user_instructions_combine_path, 'r') as f:
            user_constraints_combine = f.read()

        # iterate through each file in the profile directory
        for file in tqdm(os.listdir(profile_dir), desc='User Profile (Schema)', total=len(os.listdir(profile_dir))):
            profile_file_path = os.path.join(profile_dir, file)
            # profile data
            with open(profile_file_path, 'r') as f:
                profile_data = json.load(f)
            
            # output file path
            output_file_path = os.path.join(user_profile_output_dir, file)

            # check if the output file already exists
            if os.path.exists(output_file_path):
                # read the output file
                with open(output_file_path, 'r') as f:
                    user_profile_response = json.load(f)
            else:
                user_profile_response = []
            
            # open user sheet response
            user_sheet_response_path = os.path.join(user_sheets_output_dir, file)
            try:
                with open(user_sheet_response_path, 'r') as f:
                    user_sheet_response = json.load(f)
            except Exception as e:
                continue
            
            # iterate through each example in the profile data
            for ectr, example in tqdm(enumerate(profile_data), desc=f'Processing {file}', total=len(profile_data)):
                # check if the example already exists in the user sheet response
                if ectr < len(user_profile_response):
                    continue

                # # break after 3 iterations
                # if ectr > 2:
                #     break

                # if ectr == 0 just use the user sheet response
                if ectr == 0:
                    user_profile_response.append(user_sheet_response[ectr])
                else:
                    # construct the prompt
                    if ectr == 1:
                        prev_key = 'user_writing_sheet'
                    else:
                        prev_key = 'combined_user_sheet'
                    try:
                        prev_prompt_dict = {'previous_combined_user_sheet': extract_writing_sheet(user_profile_response[ectr - 1], key=prev_key)}
                        current_prompt_dict = {'current_writing_prompt': example['writing_prompt'], 'current_user_sheet': extract_writing_sheet(user_sheet_response[ectr], key='user_writing_sheet')}
                        
                        # construct the prompt
                        prompt = construct_user_profile_prompt(prev_prompt_dict, current_prompt_dict)
                        # call the OpenAI model
                        response = prompt_openai(prompt, max_tokens=4096, temperature=0.0)
                    except Exception as e:
                        response = user_profile_response[ectr - 1]
    
                    user_profile_response.append(response)

                # write the results to the output directory
                with open(output_file_path, 'w') as f:
                    json.dump(user_profile_response, f, indent=4)
    
        # NOTE: STEP 3: Generate story rules for each writing prompt in the test data
        test_dir = f'{self.data_split_dir}/{source}/test'

        # story rules output directory
        story_rules_output_dir = f'{self.story_rules_dir}/schema/{source}'
        if not os.path.exists(story_rules_output_dir):
            os.makedirs(story_rules_output_dir)
        
        # system instructions
        system_instructions_story_rules_path = f'{self.user_profile_instructions_dir}/system_prompts/rules.txt'
        # user instructions
        user_instructions_story_rules_path = f'{self.user_profile_instructions_dir}/user_prompts/rules.txt'

        # read the system instructions
        with open(system_instructions_story_rules_path, 'r') as f:
            system_instructions_story_rules = f.read()
        
        # read the user instructions
        with open(user_instructions_story_rules_path, 'r') as f:
            user_constraints_story_rules = f.read()
        
        # iterate through each file in the test directory
        for fctr, file in tqdm(enumerate(os.listdir(test_dir)), desc='Story Rules (Schema)', total=len(os.listdir(test_dir))):

            test_file_path = os.path.join(test_dir, file)
            # test data
            with open(test_file_path, 'r') as f:
                test_data = json.load(f)
            
            # output file path
            output_file_path = os.path.join(story_rules_output_dir, file)

            # check if the output file already exists
            if os.path.exists(output_file_path):
                # read the output file
                with open(output_file_path, 'r') as f:
                    story_rules_response = json.load(f)
            else:
                story_rules_response = []
            
            
            try:
                # open user profile response
                user_profile_response_path = os.path.join(user_profile_output_dir, file)
                with open(user_profile_response_path, 'r') as f:
                    user_profile_response = json.load(f)

                user_profile = extract_writing_sheet(user_profile_response[-1], key='combined_user_sheet')
            except Exception as e:
                continue
            
            # iterate through each example in the test data
            for ectr, example in enumerate(test_data):
                # check if the example already exists in the story rules response
                if ectr < len(story_rules_response):
                    continue
                
                if debug:
                    # break after 2 iterations
                    if ectr > 1:
                        break

                # construct the prompt
                prompt = construct_story_rules_prompt(example['writing_prompt'], user_profile)
                # call the OpenAI model
                try:
                    response = prompt_openai(prompt, max_tokens=4096, temperature=0.0)
                except Exception as e:
                    response = None
                story_rules_response.append(response)

                # write the results to the output directory
                with open(output_file_path, 'w') as f:
                    json.dump(story_rules_response, f, indent=4)
    

        # NOTE: STEP 4: Generate stories using the user profile generated above
        if few_shot_top_k != 1:
            suffix = f'_{few_shot_top_k}'
        else:
            suffix = ''

        story_output_dir = f'{self.output_dir}/schema{suffix}/{source}'
        if not os.path.exists(story_output_dir):
            os.makedirs(story_output_dir)
        
        # system instructions
        system_instructions_path = f'{self.generate_story_user_profile_instructions_dir}/system_prompts/{source}.txt'
        # read the system instructions
        with open(system_instructions_path, 'r') as f:
            system_instructions = f.read()
        
        print('Method: User Profile (Schema) Story Generation')
        print(f'Few Shot: True')
        print(f'Source: {source}')
    
        self.perform_story_generation(source=source, few_shot=True, story_output_dir=story_output_dir, source_constraints_dir = story_rules_output_dir, system_instructions=system_instructions, debug=debug, few_shot_top_k=few_shot_top_k)
    
    def rule_generator(self, source='Reddit', is_profile=False, debug=False):
        '''
        generate rules for the stories in the profile set
        '''
        def construct_rule_extractor_prompt(system_instructions, writing_prompt, profile_story, base_story, user_constraints):
            '''
            Construct the Rule Extractor Prompt
            '''
            # construct the user instruction
            user_instruction_dict = {'writing_prompt': writing_prompt, 'ground_truth_story': profile_story, 'base_story': base_story}
            user_instruction = f"{json.dumps(user_instruction_dict, indent=4)}\n\n"

            # construct OpenAI prompt
            prompt = construct_prompt_message(system_instructions, user_instruction, user_constraints)
            return prompt



        # ground truth profile directory
        if is_profile:
            consider_dir = f'{self.data_split_dir}/{source}/profile'
        else:
            consider_dir = f'{self.data_split_dir}/{source}/test'

        # base story directory 
        if is_profile:
            base_story_dir = f'{self.output_dir_profile}/vanilla/{source}'
        else:
            base_story_dir = f'{self.output_dir}/vanilla/{source}'

        # story rules output directory
        if is_profile:
            story_rules_output_dir = f'{self.output_dir_profile}/rules/{source}'
        else:
            story_rules_output_dir = f'{self.output_dir}/oracle_rules/{source}'
        if not os.path.exists(story_rules_output_dir):
            os.makedirs(story_rules_output_dir)

        # system instructions
        with open(f"{self.rule_extractor_instructions_dir}/system_prompts/delta_rules.txt", 'r') as f:
            system_instructions = f.read()
        
        # user instructions
        with open(f"{self.rule_extractor_instructions_dir}/user_prompts/delta_rules.txt", 'r') as f:
            user_constraints = f.read()
        
        # iterate through each file in the profile directory
        for fctr, file in tqdm(enumerate(os.listdir(consider_dir)), desc='Rule Generation', total=len(os.listdir(consider_dir))):
            if debug:
                # break after 3 iterations
                if fctr > 2:
                    break

            data_file_path = os.path.join(consider_dir, file)
            # profile data
            with open(data_file_path, 'r') as f:
                human_data = json.load(f)
            
            # base story file path
            base_story_file_path = os.path.join(base_story_dir, file)
            # base story data
            try:
                with open(base_story_file_path, 'r') as f:
                    base_story_data = json.load(f)
            except Exception as e:
                continue
            
            # output file path
            output_file_path = os.path.join(story_rules_output_dir, file)

            # check if the output file already exists
            if os.path.exists(output_file_path):
                # read the output file
                with open(output_file_path, 'r') as f:
                    story_rules_response = json.load(f)
            else:
                story_rules_response = []
            
            # iterate through each example in the profile data
            for ectr, example in tqdm(enumerate(human_data), desc=f'Processing {file}', total=len(human_data)):
                # check if the example already exists in the story rules response
                if ectr < len(story_rules_response):
                    continue

                # break after 3 iterations
                if debug:
                    if ectr > 2:
                        break

                # writing prompt 
                writing_prompt = example['writing_prompt']
                # base story
                base_story = base_story_data[ectr]['story']
                # profile story
                profile_story = example['story']
                # construct the prompt
                prompt = construct_rule_extractor_prompt(system_instructions, writing_prompt, profile_story, base_story, user_constraints)
                # call the OpenAI model
                try:
                    response = prompt_openai(prompt, max_tokens=4096, temperature=0.0)
                except Exception as e:
                    response = None
                story_rules_response.append(response)

                # write the results to the output directory
                with open(output_file_path, 'w') as f:
                    json.dump(story_rules_response, f, indent=4)
    
    def personalized_rule_generator(self, source='Reddit', debug=False, few_shot_top_k=1):
        '''
        generate personalized rules for the stories in the test set
        '''
        def construct_personalized_story_rules_prompt(system_instructions, writing_prompt, few_shot_examples, user_constraints):
            '''
            Construct the Rule Generator Prompt
            '''
            # construct the user instruction
            user_instruction = f"Writing Prompt: {writing_prompt}"

            # construct OpenAI prompt
            prompt = construct_prompt_message(system_instructions, user_instruction, user_constraints, few_shot_examples, add_at_end=True)
            return prompt

        # ground truth profile directory
        profile_dir = f'{self.data_split_dir}/{source}/profile'

        # test directory
        test_dir = f'{self.data_split_dir}/{source}/test'

        # profile story rules output directory
        profile_story_rules_input_dir = f'{self.output_dir_profile}/rules/{source}'

        # system instructions
        with open(f"{self.rule_extractor_instructions_dir}/system_prompts/rule_generator.txt", 'r') as f:
            system_instructions = f.read()
        
        # user instructions
        with open(f"{self.rule_extractor_instructions_dir}/user_prompts/rule_generator.txt", 'r') as f:
            user_constraints = f.read()
        
        # story rules output directory
        story_rules_output_dir = f'{self.story_rules_dir}/delta/{source}'
        if not os.path.exists(story_rules_output_dir):
            os.makedirs(story_rules_output_dir)

        # iterate through each file in the test directory
        for fctr, file in tqdm(enumerate(os.listdir(test_dir)), desc='Story Rules (Schema)', total=len(os.listdir(test_dir))):
                        
            profile_file_path = os.path.join(profile_dir, file)
            test_file_path = os.path.join(test_dir, file)

            # profile data
            with open(profile_file_path, 'r') as f:
                profile_data = json.load(f)
            
            # test data
            with open(test_file_path, 'r') as f:
                test_data = json.load(f)
            
            # output file path
            output_file_path = os.path.join(story_rules_output_dir, file)

            # check if the output file already exists
            if os.path.exists(output_file_path):
                # read the output file
                with open(output_file_path, 'r') as f:
                    story_rules_response = json.load(f)
            else:
                story_rules_response = []
            
            try:
                # open profile_story_rules_input_dir
                profile_story_rules_path = os.path.join(profile_story_rules_input_dir, file)
                with open(profile_story_rules_path, 'r') as f:
                    profile_story_rules = json.load(f)
            except Exception as e:
                continue
            
            # iterate through each example in the test data
            for ectr, example in tqdm(enumerate(test_data), desc=f'Processing {file}', total=len(test_data)):
                # check if the example already exists in the story rules response
                if ectr < len(story_rules_response):
                    continue
                
                if debug:
                    # break after 2 iterations
                    if ectr > 1:
                        break
                
                _, profile_indices = self.get_few_shot_examples(profile_data, example, source=source, top_k=3)

                # construct few shot examples
                few_shot_examples = {}
                for pctr, pindex in enumerate(profile_indices):
                    # construct user instruction 
                    profile_rules_raw = profile_story_rules[pindex]
                    # extract text between <story_rules></story_rules> tag
                    profile_rules = re.search(r'<story_rules>(.*?)</story_rules>', profile_rules_raw, re.DOTALL).group(1)
                    if not profile_rules:
                        profile_rules = profile_rules_raw
                    few_shot_examples[pctr] = {'User': f'Writing Prompt: {profile_data[pindex]['writing_prompt']}', 'Assistant': f'Story Rules: {profile_rules}'}


                # construct the prompt
                prompt = construct_personalized_story_rules_prompt(system_instructions, example['writing_prompt'], few_shot_examples, user_constraints)
                # call the OpenAI model
                try:
                    response = prompt_openai(prompt, max_tokens=4096, temperature=0.0)
                except Exception as e:
                    response = None
                story_rules_response.append(response)

                # write the results to the output directory
                with open(output_file_path, 'w') as f:
                    json.dump(story_rules_response, f, indent=4)
    
        # TODO: Story Generation using the personalized rules
        if few_shot_top_k != 1:
            suffix = f'_{few_shot_top_k}'
        else:
            suffix = ''

        story_output_dir = f'{self.output_dir}/delta{suffix}/{source}'

        if not os.path.exists(story_output_dir):
            os.makedirs(story_output_dir)
        
        # system instructions
        system_instructions_path = f'{self.generate_story_user_profile_instructions_dir}/system_prompts/{source}.txt'
        # read the system instructions
        with open(system_instructions_path, 'r') as f:
            system_instructions = f.read()
        
        print('Method: Personalized Rule Generator Story Generation')
        print(f'Few Shot: True')
        print(f'Source: {source}')

        self.perform_story_generation(source=source, few_shot=True, story_output_dir=story_output_dir, source_constraints_dir = story_rules_output_dir, system_instructions=system_instructions, debug=debug)

    def perform_oracle(self, source='Reddit', debug=False):
        '''
        Oracle Story Generation for the test set
        '''
    
        story_output_dir = f'{self.output_dir}/oracle/{source}'
        if not os.path.exists(story_output_dir):
            os.makedirs(story_output_dir)
        
        # system instructions
        system_instructions_path = f'{self.generate_story_user_profile_instructions_dir}/system_prompts/{source}.txt'
        # read the system instructions
        with open(system_instructions_path, 'r') as f:
            system_instructions = f.read()
        
        source_constraints_dir = f'{self.output_dir}/oracle_rules/{source}'
        
        print('Method: Oracle Story Generation')
        print(f'Few Shot: True')
        print(f'Source: {source}')
    
        self.perform_story_generation(source=source, few_shot=True, story_output_dir=story_output_dir, source_constraints_dir = source_constraints_dir, system_instructions=system_instructions, debug=debug)



def parse_args():
    parser = argparse.ArgumentParser(description='Story Generation Methods')
    # few shot
    parser.add_argument('--few_shot', action='store_true', help='Few Shot Story Generation')
    # few shot top k (int)
    parser.add_argument('--few_shot_top_k', type=int, default=1, help='Few Shot Top K')
    # source
    parser.add_argument('--source', type=str, default='Reddit', help='Source of the data')
    # # user profile (no schema) 'store_true'
    # parser.add_argument('--no_schema', action='store_true', help='User Profile (No Schema)')
    # int choice
    parser.add_argument('--choice', type=int, default=1, help='Choice of the method: 1. Vanilla, 2. User Profile (No Schema) 3. User Profile (Schema), 4. Personaized Rule Generator, 5. Oracle')
    # is_profile
    parser.add_argument('--is_profile', action='store_true', help='generate on profile data')
    # extract rules
    parser.add_argument('--extract_rules', action='store_true', help='extract rules')
    # debug mode
    parser.add_argument('--debug', action='store_true', help='Debug Mode')
    return parser.parse_args()

def main():
    # parse arguments
    args = parse_args()
    # source
    source = args.source    
    # few shot 
    few_shot = args.few_shot
    # few shot top k
    few_shot_top_k = args.few_shot_top_k
    # few_shot_top_k = 3
    # # method choice
    choice = args.choice
    # choice = 3
    # is_profile
    is_profile = args.is_profile
    # # extract rules
    extract_rules = args.extract_rules
    # create an instance of the StoryGenMethods class
    story_gen_methods = StoryGenMethods()

    if extract_rules:
        # extract rules
        story_gen_methods.rule_generator(source=source, is_profile=is_profile, debug=args.debug)
    else:
        if choice == 1:
            # perform Vanilla story generation
            story_gen_methods.perform_vanilla(source=source, few_shot=few_shot, debug=args.debug, is_profile=is_profile)
        elif choice == 2:
            # User Profile (No Schema)
            story_gen_methods.no_schema_user_profile(source=source, debug=args.debug)
        elif choice == 3:
            # User Profile (Schema)
            story_gen_methods.schema_user_profile(source=source, debug=args.debug, few_shot_top_k=few_shot_top_k)
        elif choice == 4:
            # Rule Generator
            story_gen_methods.personalized_rule_generator(source=source, debug=args.debug, few_shot_top_k=few_shot_top_k)
        elif choice == 5:
            # oracle generator
            story_gen_methods.perform_oracle(source=source, debug=args.debug)

if __name__ == '__main__':
    main()
            

            
