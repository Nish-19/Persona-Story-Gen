"""
utility function for prompting OpenAI models
"""

import os
from openai import OpenAI, AzureOpenAI
import time
from tenacity import retry, wait_random_exponential, stop_after_attempt


def construct_prompt_message(system_prompt, user_prompt, few_shot_prompt=None):
    """
    Construct a prompt message for OpenAI ChatCompletion model
    """
    prompt_message = []
    prompt_message.append({"role": "system", "content": system_prompt})
    if few_shot_prompt:
        for example_num, example in few_shot_prompt.items():
            prompt_message.append({"role": "user", "content": example["User"]})
            prompt_message.append(
                {"role": "assistant", "content": example["Assistant"]}
            )

    prompt_message.append({"role": "user", "content": user_prompt})

    return prompt_message


# AzureOpenAI
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def prompt_openai(prompt_messages, max_tokens=2000, temperature=1.0, top_p=0.95):
    # client = AzureOpenAI(
    #     # api_key=os.environ("AZURE_OPENAI_API_KEY"),
    #     api_version="2024-02-01",
    #     azure_endpoint = "https://complicated.openai.azure.com"
    # )

    client = OpenAI()

    completion = client.chat.completions.create(
        # model='4o',
        model="gpt-4o-2024-11-20",
        messages=prompt_messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content
