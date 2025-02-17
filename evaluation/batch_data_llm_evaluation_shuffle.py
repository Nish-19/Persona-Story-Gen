"""
OPENAI Batch API data creation
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


def construct_compare_prompt_message(
    gt_wp, gt_story, story_a, story_b, system_prompt, user_constraints, cat, cat_value
):
    """
    construct prompt for pair-wise comparison
    """
    # check if gt_story is dict
    input_dict = {
        "Writing Prompt": gt_wp,
        "Human-Written Story": gt_story,
        "Assistant A": story_a,
        "Assistant B": story_b,
        "**Specified Storytelling Aspect to Evaluate** (evaluate on this aspect only)": f"{cat}: {cat_value}",
    }

    user_instruction = f"{json.dumps(input_dict)}"
    # NOTE: Replace <Fill Here> in user_instruction with cat values
    user_constraints = user_constraints.replace("<Fill Here>", f"{cat}: {cat_value}")
    prompt = construct_prompt_message(system_prompt, user_instruction, user_constraints)
    return prompt


def create_openai_batch_input(prompt, identifier):
    """
    create batch input for OpenAI
    """
    task = {
        "custom_id": identifier,  # custom_id must be a string
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 2000,
            "messages": prompt,
        },
    }

    return task


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
    output_dir = f"batch_llm_evaluation_shuffle_score{llama_suffix}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = f"{output_dir}/{consider_dir}.jsonl"

    # read compare prompts
    system_prompt_path = f"instructions/system_prompt/compare_score.txt"
    user_constraints_path = f"instructions/user_prompt/compare_score.txt"
    categories_path = "instructions/user_prompt/compare_categories.json"

    # NOTE: 1. Compare prompts
    # read the system prompt
    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()
    # read the user constraints
    with open(user_constraints_path, "r") as f:
        user_constraints = f.read()

    # read the categories
    with open(categories_path, "r") as f:
        categories_data = json.load(f)
    # define the categories
    categories = categories_data.keys()

    sources = ["Reddit", "AO3", "Storium", "narrativemagazine", "newyorker"]

    # batch data list to store the data
    batch_data_list = []

    # iterate over sources
    for source in sources:
        gt_root_dir = f"../datasets/data_splits/data/{source}/test/"
        expts_root_dir = f"../experiments/results{llama_suffix}/{consider_dir}/{source}"
        pairs = []
        # NOTE: Construct evaluation pairs
        # iterate over files in the ground truth directory
        for file in tqdm(
            os.listdir(gt_root_dir),
            desc="Processing Authors",
            total=len(os.listdir(gt_root_dir)),
        ):
            # gt file path
            gt_file_path = os.path.join(gt_root_dir, file)
            # vanilla file path
            vanilla_file_path = os.path.join(
                f"../experiments/results{llama_suffix}/vanilla/{source}", file
            )
            # expts file path
            expts_file_path = os.path.join(expts_root_dir, file)

            # read the ground truth file
            with open(gt_file_path, "r") as f:
                gt_data = json.load(f)

            try:
                # read the vanilla file
                with open(vanilla_file_path, "r") as f:
                    vanilla_data = json.load(f)

                # read the expts file
                with open(expts_file_path, "r") as f:
                    expts_data = json.load(f)
            except:
                if verbose:
                    print("Skipping", file)
                continue

            # iterrate only over expts_data
            for ectr, expts in enumerate(expts_data):
                # add the pair
                identifier = f"{source}_{file}_{ectr}"

                gt_wp = gt_data[ectr]["writing_prompt"]
                gt_story = gt_data[ectr]["story"]
                if gt_story is None or expts["story"] is None:
                    print("Skipping None", file)
                    continue
                pairs.append(
                    (
                        identifier,
                        gt_wp,
                        gt_story,
                        vanilla_data[ectr]["story"],
                        expts["story"],
                    )
                )

        print(f"Using {consider_dir} method")
        print(f"Consider {len(pairs)} pairs for comparison")

        # iterate over the pairs
        for pair in tqdm(pairs, desc="Creating Batch file data", total=len(pairs)):
            identifier, gt_wp, gt_story, vanilla_story, expts_story = pair

            for cat in categories:
                # generate random number (0 or 1)
                random_number = random.randint(0, 1)
                if random_number == 0:
                    prompt = construct_compare_prompt_message(
                        gt_wp,
                        gt_story,
                        vanilla_story,
                        expts_story,
                        system_prompt,
                        user_constraints,
                        cat,
                        categories_data[cat],
                    )
                    cur_identifier = f"{identifier}_{cat}_vanilla"
                else:
                    # reverse the order of the stories
                    prompt = construct_compare_prompt_message(
                        gt_wp,
                        gt_story,
                        expts_story,
                        vanilla_story,
                        system_prompt,
                        user_constraints,
                        cat,
                        categories_data[cat],
                    )
                    cur_identifier = f"{identifier}_{cat}_expts"
                # NOTE: construct batch prompt
                batch_input = create_openai_batch_input(prompt, cur_identifier)
                # add to the list
                batch_data_list.append(batch_input)

    print("Total number of batch data:", len(batch_data_list))

    # Add to JSONL file
    with open(output_file, "w") as f:
        for data in batch_data_list:
            f.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    main()
