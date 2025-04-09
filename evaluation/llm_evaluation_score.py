"""
Use LLMs for pair-wise comparison of the methods (random shuffle)
"""

import os
import sys
import re
import json
import argparse
from tqdm import tqdm
import time
import random
from collections import defaultdict
# from rank_bm25 import BM25Okapi
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
from prompt_llm_utils import (
    construct_prompt_message,
    prompt_openai,
    prompt_llama,
    prompt_llama_router,
    prompt_llama,
    load_prometheus_eval_model,
)


def parse_args():
    parser = argparse.ArgumentParser()
    # few shot
    parser.add_argument(
        "--few_shot", action="store_true", help="Few Shot Story Generation"
    )
    # few shot top k (int)
    parser.add_argument("--few_shot_top_k", type=int, default=1, help="Few Shot Top K")

    # source
    parser.add_argument("--source", type=str, default="Reddit", help="Source")
    # method choice
    parser.add_argument(
        "--choice",
        type=int,
        default=1,
        help="Choice of the method: 1. Vanilla, 2. User Profile (No Schema) 3. User Profile (Schema), 4. Personaized Rule Generator, 5. User Profile (Delta), 6. Oracle",
    )
    # model choice
    parser.add_argument(
        "--model_choice",
        type=int,
        default=1,
        help="Choice of the Model: 1. GPT-4o, 2. LLama-3.1-70B, 3. GPT-4o-mini, 4. Prometheus",
    )
    # history (store_true)
    parser.add_argument(
        "--history",
        action="store_true",
        help="Evaluate on Past History as compared to the ground truth",
    )
    # persona mode
    parser.add_argument(
        "--persona",
        action="store_true",
        help="To use persona prompt obtained from Author Sheet (for Schema and Delta Schema only)",
    )

    # ft baseline 
    parser.add_argument(
        "--ft_baseline",
        action="store_true",
        help="Whether to use the fine-tuned baseline model as Average Author",
    )

    # verbose (store_true)
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    # azure (store_true)
    parser.add_argument("--azure", action="store_true", help="To use azure openai")
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
    if isinstance(gt_story, dict):
        input_dict = {
            "Author Style Summary": gt_story,
            "Writing Prompt": gt_wp,
            "Assistant A": story_a,
            "Assistant B": story_b,
            "Specified Storytelling Aspect to Evaluate": f"{cat}: {cat_value}",
        }
    # elif isinstance(gt_story, list):
    #     input_dict = {'Author History': gt_story, 'New Writing Prompt': gt_wp, 'Assistant A': story_a, 'Assistant B': story_b}
    else:
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

def construct_prometheus_prompt(wp, gt_story, cat, cat_value):
    '''
    Create a prompt for the faithfulness evaluation.
    '''

    instruction = (
        f"Write a story in response to the following writing prompt:\n\n"
        f"{wp}\n\n"
        f"Your story should follow the style to the following Human-Written Story with respect to the story-writing aspect, '{cat}': {cat_value}.\n\n"
        f"Human-Written Story:\n{gt_story}"
    )

    # instruction = (
    #     f"Write a story in response to the following prompt:\n\n"
    #     f"{wp}\n\n"
    #     f"Your story should be similar in style to the story written below as the Reference Answer with respect to the following story-writing aspect, '{cat}': {cat_value}.\n\n"
    #     "Do not consider any other story-writing aspect other than the one mentioned above for evaluation."
    # )

    rubric = (
        f"Is the story response similar in style to the Human-Written story with respect to the specified story-writing aspect, '{cat}'? "
        f"Do not evaluate the story on its overall quality or in isolation — focus solely on how well it aligns with the Human-Written story "
        f"in terms of the given aspect."
    )

    return instruction, rubric



def construct_summarize_prompt_message(
    history_data, system_prompt, user_constraints, cat, cat_value
):
    """
    construct prompt for pair-wise comparison
    """
    # check if gt_story is dict
    input_dict = {"Author History": history_data}

    user_instruction = f"{json.dumps(input_dict)}"
    # NOTE: Replace <Fill Here> in user_instruction with cat values
    user_constraints = user_constraints.replace("<Fill Here>", f"{cat}: {cat_value}")
    prompt = construct_prompt_message(system_prompt, user_instruction, user_constraints)
    return prompt


def get_few_shot_indices(profile_data, example, top_k=1):
    """
    return the few shot examples
    """
    # get most similar examples from the profile data using BM25
    profile_prompts = [p["writing_prompt"] for p in profile_data]
    query = example["writing_prompt"]

    # Tokenize the prompts and query
    stop_words = set(stopwords.words("english"))
    tokenized_prompts = [
        [word for word in word_tokenize(prompt.lower()) if word not in stop_words]
        for prompt in profile_prompts
    ]
    tokenized_query = [
        word for word in word_tokenize(query.lower()) if word not in stop_words
    ]

    # Perform BM25
    bm25 = BM25Okapi(tokenized_prompts)
    scores = bm25.get_scores(tokenized_query)
    profile_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
        :top_k
    ]

    return profile_indices


def main():
    # parse arguments
    args = parse_args()

    # set random seed
    random.seed(37)

    # few shot
    few_shot = args.few_shot
    # few shot top k
    few_shot_top_k = args.few_shot_top_k

    # source
    source = args.source
    # choice
    choice = args.choice
    # model choice
    model_choice = args.model_choice
    # history
    history = args.history
    # persona
    persona = args.persona
    # azure
    azure = args.azure
    # llama
    llama = args.llama
    # llama70
    llama70 = args.llama70
    # ft_baseline
    ft_baseline = args.ft_baseline

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

    # history
    if history:
        his_suffix = "_summarize_history"
    else:
        his_suffix = ""

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

    if source == "all":
        sources = ["Reddit", "AO3", "narrativemagazine", "newyorker", "Storium"]
    else:
        sources = [source]

    # ft_baseline data
    if ft_baseline:
        print("Using FT Baseline")
        ft_baseline_data_dir = '../experiments/finetune/sft-8b-no-len/test_results.json'
        # load the data
        with open(ft_baseline_data_dir, "r") as f:
            ft_baseline_raw_data = json.load(f)
        
        ft_baseline_data = {}
        # process the data
        for data in ft_baseline_raw_data:
            # remove the first element
            ft_baseline_data[data['wp']] = data['pred_story']
        
        # set ft_flag
        ft_flag = "_ft_baseline"
    else:
        ft_flag = ""

    if model_choice == 4:
        # load the prometheus model
        prometheus_judge = load_prometheus_eval_model()

    # iterate over sources
    for source in sources:
        print(f"### Processsing {source} ###")
        # root directories
        gt_root_dir = f"../datasets/data_splits/data/{source}/test/"
        story_root_dir = f"../datasets/{source}/selected_human_with_prompts/"
        profile_root_dir = f"../datasets/data_splits/data/{source}/profile/"

        expts_root_dir = f"../experiments/results{llama_suffix}/{consider_dir}/{source}"

        # results output directory
        output_dir = f"llm_evaluation_shuffle_score{his_suffix}{llama_suffix}/{consider_dir}/{model_choice}{ft_flag}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = f"{output_dir}/{source}.json"
        # check if the file exists
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                all_responses = json.load(f)
            # convert to defaultdict
            all_responses = defaultdict(dict, all_responses)
        else:
            all_responses = defaultdict(dict)

        # check history
        if history:
            summarize_out_dir = (
                f"llm_evaluation_shuffle_score{his_suffix}/{consider_dir}/1"
            )
            output_summarize_history_dir = f"{summarize_out_dir}/summarize_history"
            if not os.path.exists(output_summarize_history_dir):
                os.makedirs(output_summarize_history_dir)
            output_summarize_history_path = (
                f"{output_summarize_history_dir}/{source}.json"
            )
            # check if the file exists
            if os.path.exists(output_summarize_history_path):
                with open(output_summarize_history_path, "r") as f:
                    all_responses_summarize_history = json.load(f)
            else:
                all_responses_summarize_history = {}

        # read compare prompts
        system_prompt_path = f"instructions/system_prompt/compare_score{his_suffix}.txt"
        user_constraints_path = (
            f"instructions/user_prompt/compare_score{his_suffix}.txt"
        )

        # read summarize history prompts
        system_prompt_sumhis_path = "instructions/system_prompt/summarize_history.txt"
        user_constraints_sumhis_path = "instructions/user_prompt/summarize_history.txt"

        categories_path = "instructions/user_prompt/compare_categories.json"

        # NOTE: 1. Compare prompts
        # read the system prompt
        with open(system_prompt_path, "r") as f:
            system_prompt = f.read()
        # read the user constraints
        with open(user_constraints_path, "r") as f:
            user_constraints = f.read()

        # NOTE: 2. Summarize History prompts
        # read the system prompt
        with open(system_prompt_sumhis_path, "r") as f:
            system_prompt_sumhis = f.read()
        # read the user constraints
        with open(user_constraints_sumhis_path, "r") as f:
            user_constraints_sumhis = f.read()

        # read the categories
        with open(categories_path, "r") as f:
            categories_data = json.load(f)

        # define the categories
        categories = categories_data.keys()

        pairs = []
        
        # prepare for batch relative grade
        if model_choice == 4:
            # batch relative grade
            instructions = []  # List of instructions
            responses_from_a = []  # List of responses
            responses_from_b = []
            # rubric = "Is the story similar to the Human-Written Story for the story-writing aspect?"
            rubrics = []
            reference_answers = []  # List of reference answers
            # extra
            identifiers = []  # List of identifiers
            markers = []  # List of markers
            eval_categories = []  # List of categories


        # iterate over files in the ground truth directory
        for file in tqdm(
            os.listdir(gt_root_dir),
            desc="Processing Authors",
            total=len(os.listdir(gt_root_dir)),
        ):
            # gt file path
            gt_file_path = os.path.join(gt_root_dir, file)
            # story file path
            story_file_path = os.path.join(story_root_dir, file)
            # profile file path
            profile_file_path = os.path.join(profile_root_dir, file)
            # vanilla file path
            vanilla_file_path = os.path.join(
                f"../experiments/results{llama_suffix}/vanilla/{source}", file
            )
            # expts file path
            expts_file_path = os.path.join(expts_root_dir, file)

            # read the ground truth file
            with open(gt_file_path, "r") as f:
                gt_data = json.load(f)
            
            # read the story file
            with open(story_file_path, "r") as f:
                story_data_raw = json.load(f)
            
            # process the story data
            story_data = {}
            for data in story_data_raw:
                story_data[data['writing_prompt']] = data['comment']

            # read the profile file
            with open(profile_file_path, "r") as f:
                profile_data = json.load(f)

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

            if history:
                # last_story_wp = profile_data[-1]['writing_prompt']
                # last_story = profile_data[-1]['story']
                # last_story_data = {'writing_prompt': last_story_wp, 'story': last_story}
                # get all the history data
                history_data = [
                    {"writing_prompt": p["writing_prompt"], "story": p["story"]}
                    for p in profile_data
                ]

                # TODO: summarize history for each category
                # 1. Check if file exists in all_responses_summarize_history
                if file in all_responses_summarize_history:
                    summarize_history = all_responses_summarize_history[file]
                else:
                    summarize_history = {}
                    for cat in categories:
                        # construct the prompt
                        prompt = construct_summarize_prompt_message(
                            history_data,
                            system_prompt_sumhis,
                            user_constraints_sumhis,
                            cat,
                            categories_data[cat],
                        )
                        # prompt the OpenAI model
                        if model_choice == 1:
                            response = prompt_openai(
                                prompt, model="gpt-4o", azure=azure
                            )
                        elif model_choice == 2:
                            response = prompt_llama_router(prompt)
                            # response = prompt_llama(prompt)
                        elif model_choice == 3:
                            response = prompt_openai(
                                prompt, model="gpt-4o-mini", azure=azure
                            )
                        # extract response in the tags <analysis></analysis>
                        response_match = re.search(
                            r"<analysis>(.*)</analysis>", response, re.DOTALL
                        )
                        if response_match:
                            response = response_match.group(1)
                        else:
                            response = response

                        summarize_history[cat] = response

                        # add the responses to the dictionary
                        all_responses_summarize_history[file] = summarize_history
                        # write the responses to a file
                        with open(output_summarize_history_path, "w") as f:
                            json.dump(all_responses_summarize_history, f, indent=4)
            # else:
            #     last_story_data = None

            # iterrate only over expts_data
            for ectr, expts in enumerate(expts_data):
                # add the pair
                identifier = f"{file}_{ectr}"

                # check if the identifier exists in the output file
                if identifier in all_responses:
                    if verbose:
                        print(f"Skipping {identifier}")
                    continue

                gt_wp = gt_data[ectr]["writing_prompt"]
                gt_story = gt_data[ectr]["story"]
                # gt_story = story_data.get(gt_wp, None)
                if gt_story is None or expts["story"] is None:
                    print("Skipping None", file)
                    continue
                
                average_author = vanilla_data[ectr]["story"] if not ft_baseline else ft_baseline_data[gt_wp]

                if history:
                    pairs.append(
                        (
                            identifier,
                            gt_wp,
                            summarize_history,
                            average_author,
                            expts["story"],
                        )
                    )
                else:
                    pairs.append(
                        (
                            identifier,
                            gt_wp,
                            gt_story,
                            average_author,
                            expts["story"],
                        )
                    )

        print(f"Using {consider_dir} method")
        print(f"Consider {len(pairs)} pairs for comparison")

        # iterate over the pairs
        for pair in tqdm(pairs, desc="Pair-wise Evaluation", total=len(pairs)):
            identifier, gt_wp, gt_story, vanilla_story, expts_story = pair

            cat_dict = {}
            for cat in categories:
                # check type of gt_story
                if isinstance(gt_story, dict):
                    gt_story_input = {cat: gt_story[cat]}
                else:
                    gt_story_input = gt_story

                # generate random number (0 or 1)
                random_number = random.randint(0, 1)
                if random_number == 0:

                    prompt = construct_compare_prompt_message(
                        gt_wp,
                        gt_story_input,
                        vanilla_story,
                        expts_story,
                        system_prompt,
                        user_constraints,
                        cat,
                        categories_data[cat],
                    )
                    # prompt the OpenAI model
                    if model_choice == 1:
                        response = prompt_openai(prompt, model="gpt-4o", azure=azure)
                    elif model_choice == 2:
                        response = prompt_llama_router(prompt)
                        # response = prompt_llama(prompt)
                    elif model_choice == 3:
                        response = prompt_openai(
                            prompt, model="gpt-4o-mini", azure=azure
                        )
                        user_constraints += "Important: Please ensure to evaluate only on the specified story-telling aspect and no other."
                    elif model_choice == 4:
                        # construct prometheus prompt
                        instruction, rubric = construct_prometheus_prompt(
                            gt_wp, gt_story_input, cat, categories_data[cat])
                        
                        # append data to lists
                        instructions.append(instruction)
                        responses_from_a.append(vanilla_story)
                        responses_from_b.append(expts_story)
                        reference_answers.append(gt_story_input)
                        rubrics.append(rubric)
                        identifiers.append(identifier)
                        markers.append("A: vanilla")
                        eval_categories.append(cat)

                    # construct response dict
                    if model_choice != 4:
                        response_dict = {1: response, 2: "A: vanilla"}
                else:
                    # reverse the order of the stories
                    prompt = construct_compare_prompt_message(
                        gt_wp,
                        gt_story_input,
                        expts_story,
                        vanilla_story,
                        system_prompt,
                        user_constraints,
                        cat,
                        categories_data[cat],
                    )
                    # prompt the OpenAI model
                    if model_choice == 1:
                        response = prompt_openai(prompt, azure=azure)
                    elif model_choice == 2:
                        response = prompt_llama_router(prompt)
                        # response = prompt_llama(prompt)
                    elif model_choice == 3:
                        response = prompt_openai(
                            prompt, model="gpt-4o-mini", azure=azure
                        )
                        user_constraints += "Important: Please ensure to evaluate only on the specified story-telling aspect and no other."
                    elif model_choice == 4:
                        # construct prometheus prompt
                        instruction, rubric = construct_prometheus_prompt(
                            gt_wp, gt_story_input, cat, categories_data[cat])
                        
                        # append data to lists
                        instructions.append(instruction)
                        responses_from_a.append(expts_story)
                        responses_from_b.append(vanilla_story)
                        reference_answers.append(gt_story_input)
                        rubrics.append(rubric)
                        identifiers.append(identifier)
                        markers.append("A: expts")
                        eval_categories.append(cat)
                    
                    # construct response dict
                    if model_choice != 4:
                        response_dict = {1: response, 2: "A: expts"}
                # add the responses to the dictionary
                if model_choice != 4:
                    cat_dict[cat] = response_dict

                    # add the responses to the list
                    all_responses[identifier] = cat_dict

                    # write the responses to a file
                    with open(output_file, "w") as f:
                        json.dump(all_responses, f, indent=4)

                    # sleep for 5 seconds
                    time.sleep(5)

        if model_choice == 4:
            # TODO: Implement batch processing for Prometheus
            print("Data Prepared for Prometheus Evaluation")
            feedbacks, scores = prometheus_judge.relative_grade(
                instructions=instructions,
                responses_A=responses_from_a,
                responses_B=responses_from_b,
                reference_answers=reference_answers,
                rubric=rubrics,
            )

            # unpack and dump results
            for idx, (feedback, score) in enumerate(zip(feedbacks, scores)):
                identifier = identifiers[idx]
                cat = eval_categories[idx]
                marker = markers[idx]

                # construct response
                response = f"<evaluation>{feedback}</evaluation>\n<score>{score}</score>"

                # construct response dict
                response_dict = {1: response, 2: marker, "Category": cat}

                # add the responses to the dictionary
                all_responses[identifier][cat] = response_dict

                # write the responses to a file
                with open(output_file, "w") as f:
                    json.dump(all_responses, f, indent=4)
            
            # force reset batch information
            instructions, responses_from_a, responses_from_b = [], [], []
            # rubrics = []
            reference_answers, rubrics = [], []
            identifiers, markers, eval_categories = [], [], []



if __name__ == "__main__":
    main()
