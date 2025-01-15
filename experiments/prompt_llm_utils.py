'''
utility function for prompting OpenAI models
'''

import os
import requests
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

def prompt_llama(prompt_messages, max_tokens=2000, temperature=0.0, top_p=1.0, port_choice=1):

    if port_choice == 1:
        base_url = "http://127.0.0.1:30000/v1"
    elif port_choice == 2:
        base_url = "http://127.0.0.1:50000/v1"
    elif port_choice == 3:
        base_url = "http://127.0.0.1:10000/v1"

    client = OpenAI(base_url=base_url, api_key="None")

    response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            messages=prompt_messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    return response.choices[0].message.content

def prompt_llama_router(prompt_messages, max_tokens=2000, temperature=0.0, top_p=1.0):
    # Retrieve the OpenRouter API key
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
    
    # Define the base URL and headers
    base_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
        # Optional headers for OpenRouter rankings
        "HTTP-Referer": "https://your-site-url.com",  # Replace with your site URL
        "X-Title": "Story Generation",  # Replace with your app name
    }

    # Define the request payload
    body = {
        "model": "meta-llama/llama-3.1-70b-instruct",
        "messages": prompt_messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "provider": {
            "order": ["Fireworks", "Perplexity"],  # Provider prioritization
            "allow_fallbacks": False  # Disable fallbacks
        }
    }

    # Make the POST request
    response = requests.post(base_url, headers=headers, json=body)

    # Handle errors
    if response.status_code != 200:
        raise RuntimeError(f"Request failed with status {response.status_code}: {response.text}")

    # Return the generated response
    return response.json()["choices"][0]["message"]["content"]
