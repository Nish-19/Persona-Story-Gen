'''
utility function for prompting OpenAI models
'''

import os
from openai import OpenAI, AzureOpenAI
import time
from tenacity import retry, wait_random_exponential, stop_after_attempt


def construct_prompt_message(system_prompt, user_prompt, user_constraints=None, few_shot_prompt=None, add_at_end=False):
    '''
    Construct a prompt message for OpenAI ChatCompletion model
    '''
    prompt_message = []
    # add system prompt
    prompt_message.append({'role': 'system', 'content': system_prompt})
    # add user constraints
    if not add_at_end:
        if user_constraints:
            prompt_message.append({'role': 'user', 'content': user_constraints})
    if few_shot_prompt:
        for example_num, example in few_shot_prompt.items():
            prompt_message.append({'role': 'user', 'content': example["User"]})
            prompt_message.append({'role': 'assistant', 'content': example["Assistant"]})
    if add_at_end:
        if user_constraints:
            prompt_message.append({'role': 'user', 'content': user_constraints})

    prompt_message.append({'role': 'user', 'content': user_prompt})

    return prompt_message

# AzureOpenAI
@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(6))
def prompt_openai(prompt_messages, max_tokens=2000, temperature=1.0, top_p=1.0):
    # client = AzureOpenAI(
    #     # api_key=os.environ("AZURE_OPENAI_API_KEY"),
    #     api_version="2024-02-01",
    #     azure_endpoint = "https://complicated.openai.azure.com"
    # )

    client = OpenAI()


    completion = client.chat.completions.create(
            # model='4o',
            model='gpt-4o',
            messages=prompt_messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    
    # sleep for 1 second
    time.sleep(1)
    return completion.choices[0].message.content