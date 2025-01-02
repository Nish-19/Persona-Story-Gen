'''
utility function for prompting OpenAI models
'''

import os
from openai import OpenAI, AzureOpenAI
import time
from tenacity import retry, wait_random_exponential, stop_after_attempt


def construct_prompt_message(system_prompt, user_prompt, user_constraints=None, few_shot_prompt=None):
    '''
    Construct a prompt message for OpenAI ChatCompletion model
    '''
    prompt_message = []
    # add system prompt
    prompt_message.append({'role': 'system', 'content': system_prompt})
    # add user constraints
    if user_constraints:
        prompt_message.append({'role': 'user', 'content': user_constraints})
    if few_shot_prompt:
        for example_num, example in few_shot_prompt.items():
            prompt_message.append({'role': 'user', 'content': example["User"]})
            prompt_message.append({'role': 'assistant', 'content': example["Assistant"]})

    prompt_message.append({'role': 'user', 'content': user_prompt})

    return prompt_message

# AzureOpenAI
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def prompt_openai(prompt_messages, max_tokens=2000, temperature=1.0, top_p=1.0):
    # client = AzureOpenAI(
    #     # api_key=os.environ("AZURE_OPENAI_API_KEY"),
    #     api_version="2024-02-01",
    #     azure_endpoint = "https://complicated.openai.azure.com"
    # )

    client = OpenAI()


    completion = client.chat.completions.create(
            model='gpt-4o', # model='4o',
            messages=prompt_messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    return completion.choices[0].message.content


def prompt_llama(prompt_messages, max_tokens=2000, temperature=0.0, top_p=1.0):
    client = OpenAI(base_url="http://10.100.20.10:30000/v1", api_key="None")

    response = client.chat.completions.create(
            # model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            model="/datasets/ai/llama3/meta-llama/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/33101ce6ccc08fa6249c10a543ebfcac65173393",
            messages=prompt_messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    return response.choices[0].message.content