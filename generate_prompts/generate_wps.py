"""
Generate writing prompts for stories
"""

import os
from tqdm import tqdm
import json
import argparse
from prompt_llm_utils import construct_prompt_message, prompt_openai


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=int,
        default=1,
        help="Data Category. 1: AO3, 2: NarrativeMagazine, 3: NewYorker, 4: Reddit, 5. Storium",
    )
    return parser.parse_args()


def main():
    # args
    args = parse_args()

    # load the system prompt
    with open("resources/system_prompt.txt", "r") as f:
        system_prompt = f.read()

    # few shot prompts
    with open("resources/few_shot.json", "r") as f:
        few_shot_prompts = json.load(f)

    if args.data == 1:
        data_choice = "AO3"
    elif args.data == 2:
        data_choice = "narrativemagazine"
    elif args.data == 3:
        data_choice = "newyorker"
    elif args.data == 4:
        data_choice = "Reddit"
    elif args.data == 5:
        data_choice = "Storium"
    else:
        raise ValueError("Invalid data category. Chose 1, 2, 3, 4, or 5")

    # load data
    data_dir = f"../datasets/{data_choice}/selected_human"

    output_dir = f"../datasets/{data_choice}/selected_human_with_prompts"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # iterate through each file
    for file in tqdm(
        os.listdir(data_dir),
        desc="Generating Writing Prompts",
        total=len(os.listdir(data_dir)),
    ):
        if file.endswith(".json"):
            file_path = os.path.join(data_dir, file)
            with open(file_path, "r") as f:
                data = json.load(f)
                for i, story in enumerate(data):
                    if data_choice != "Reddit":
                        # construct prompt message
                        prompt_message = construct_prompt_message(
                            system_prompt,
                            f"Story: {story['comment']}\n\nPrompt:",
                            few_shot_prompts,
                        )
                        # prompt openai
                        completion = prompt_openai(
                            prompt_message, max_tokens=500, temperature=0.0
                        )
                        # augment the story with the completion
                        story["writing_prompt"] = completion
                    else:
                        story["writing_prompt"] = (
                            story["post_title"].split("[WP]")[1].strip()
                        )

                    # save the data after every story
                    with open(
                        f"../datasets/{data_choice}/selected_human_with_prompts/{file}",
                        "w",
                    ) as f:
                        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
