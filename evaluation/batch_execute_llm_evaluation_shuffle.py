"""
Send OpenAI request for process batch job
"""

import os
import sys
import re
import json
import argparse
from tqdm import tqdm
import time
import random
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from prompt_llm_utils import (
    construct_prompt_message,
    prompt_openai,
    prompt_llama,
    prompt_llama_router,
)
from openai import OpenAI


def parse_args():
    parser = argparse.ArgumentParser()
    # few shot
    parser.add_argument(
        "--few_shot", action="store_true", help="Few Shot Story Generation"
    )
    # few shot top k (int)
    parser.add_argument("--few_shot_top_k", type=int, default=1, help="Few Shot Top K")
    # method choice
    parser.add_argument(
        "--choice",
        type=int,
        default=1,
        help="Choice of the method: 1. Vanilla, 2. User Profile (No Schema) 3. User Profile (Schema), 4. Personaized Rule Generator, 5. User Profile (Delta), 6. Oracle",
    )
    # persona mode
    parser.add_argument(
        "--persona",
        action="store_true",
        help="To use persona prompt obtained from Author Sheet (for Schema and Delta Schema only)",
    )
    # verbose (store_true)
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    # llama (store_true)
    parser.add_argument(
        "--llama", action="store_true", help="To use llama generated model results"
    )
    # llama (store_true)
    parser.add_argument(
        "--llama70",
        action="store_true",
        help="To use llama 70B generated model results",
    )

    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()

    # set random seed
    random.seed(37)

    # few shot
    few_shot = args.few_shot
    # few shot top k
    few_shot_top_k = args.few_shot_top_k
    # choice
    choice = args.choice
    # persona
    persona = args.persona
    # llama
    llama = args.llama
    # llama70
    llama70 = args.llama70
    # verbose
    verbose = args.verbose

    # suffix
    if few_shot:
        suffix = "_few_shot"
    else:
        suffix = ""

    # persona suffix
    if persona:
        persona_suffix = "_persona"
    else:
        persona_suffix = ""

    # llama_suffix
    if llama:
        llama_suffix = "_llama"
    elif llama70:
        llama_suffix = "_llama70"
    else:
        llama_suffix = ""

    if few_shot_top_k == 1:
        top_k_suffix = ""
    else:
        top_k_suffix = f"_{few_shot_top_k}"

    # root directories
    if choice == 1:
        consider_dir = f"vanilla{suffix}"
    elif choice == 2:
        consider_dir = f"no_schema"
    elif choice == 3:
        consider_dir = f"schema{top_k_suffix}{persona_suffix}"
    elif choice == 4:
        consider_dir = f"delta{top_k_suffix}"
    elif choice == 5:
        consider_dir = f"delta_schema{top_k_suffix}{persona_suffix}"
    elif choice == 6:
        consider_dir = f"oracle{top_k_suffix}"

    # results output directory
    input_dir = f"batch_llm_evaluation_shuffle_score{llama_suffix}"
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    input_file = f"{input_dir}/{consider_dir}.jsonl"

    # Read API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    print("Using OpenAI API Key: ", api_key)
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Upload to OpenAI file API
    batch_file = client.files.create(file=open(input_file, "rb"), purpose="batch")

    print(f"Batch file uploaded to OpenAI: {batch_file.id}")

    # Start batch job
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": input_file},
    )

    print("batch_job: ", batch_job)

    # save batch job id to file
    save_dir = f"metadata_{input_dir}/{consider_dir}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = f"{save_dir}/batch_file_id.txt"
    with open(save_path, "w") as f:
        f.write(batch_file.id)

    print(f"Batch job started: {batch_job.id}")

    # save batch job
    save_path = f"{save_dir}/batch_job.txt"
    with open(save_path, "w") as f:
        f.write(str(batch_job))

    while True:
        batch_job = client.batches.retrieve(batch_job.id)
        if batch_job.status != "completed":
            time.sleep(10)
            print(f"job {batch_job.id} is still running")
        else:
            print(f"job {batch_job.id} is done")
            break


if __name__ == "__main__":
    main()
