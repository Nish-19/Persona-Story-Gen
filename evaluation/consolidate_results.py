"""
get category-wise pair-wise win-rates for each method and source
"""

import os
import argparse
import re
from collections import defaultdict
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def extract_winner(res, faith=False):
    """
    extract text between the tag <winner></winner>
    """

    def get_winner(score_text):
        # Extract scores for Story A and Story B using regex
        if faith:
            story_a_score = re.search(r"Story A:\s*(\d+)", score_text)
            story_b_score = re.search(r"Story B:\s*(\d+)", score_text)
        else:
            story_a_score = re.search(r"Assistant A:\s*(\d+)", score_text)
            story_b_score = re.search(r"Assistant B:\s*(\d+)", score_text)

        if story_a_score and story_b_score:
            score_a = int(story_a_score.group(1).strip())
            score_b = int(story_b_score.group(1).strip())

            if score_a > score_b:
                return "A"
            elif score_a < score_b:
                return "B"
            else:
                return "Tie"

        else:
            return None

    winner = None
    score_match = re.search(r"<score>(.*?)</score>", res, re.DOTALL)
    if score_match:
        score_text = score_match.group(1)
        winner = get_winner(score_text)

    elif "**score**" in res:
        score_text = res.split("**score**")[1].split("**")[0]
        winner = get_winner(score_text)

    return winner


def get_catwise_winners(source_data, faith=False):
    """
    get average win-rate for each category for the source
    """
    category_winners = defaultdict(dict)
    for key, data in source_data.items():

        # iterate over the categories
        for cat, res in data.items():
            # get labels for A and B
            label_a = res["2"].strip("A: ")
            if label_a == "vanilla":
                label_b = "expts"
            else:
                label_b = "vanilla"

            winner_label = extract_winner(res["1"], faith)

            # check if winner_label is None
            if winner_label is None:
                # count_None += 1
                # print(key, cat)
                continue

            if winner_label == "A":
                winner = label_a
            elif winner_label == "B":
                winner = label_b
            else:
                winner = "Tie"

            category_winners[cat][winner] = category_winners[cat].get(winner, 0) + 1

    # average win-rate for each category
    for cat, win_dict in category_winners.items():
        total = sum(win_dict.values())
        for key in win_dict:
            win_dict[key] /= total

    # sort category_winners based on values
    for cat, win_dict in category_winners.items():
        category_winners[cat] = {
            k: v
            for k, v in sorted(win_dict.items(), key=lambda item: item[1], reverse=True)
        }

    return category_winners


def create_graph(method_source_wise_results, output_dir):
    # Specify methods to compare
    methods_to_compare = ["oracle", "delta", "schema_persona", "delta_schema_persona"]
    # define method alias
    method_alias = {"oracle": "O", "delta_schema_persona": "WS", "schema_persona": "S", "delta": "D"}
    # source alias
    source_alias = {
        "Reddit.json": "Reddit",
        "AO3.json": "AO3",
        "Storium.json": "Storium",
        "narrativemagazine.json": "N.Magazine",
        "newyorker.json": "New Yorker",
    }

    save_dir = f"{output_dir}/graphs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define common colors for all methods
    common_colors = {"expts": "blue", "Tie": "orange", "vanilla": "green"}

    # Iterate over sources
    for source in next(iter(method_source_wise_results.values())).keys():
        if source == "overall.json":
            continue

        # Extract categories
        categories = list(
            next(iter(method_source_wise_results.values()))[source].keys()
        )
        # replace "Development (Character and Setting)" with "Development"
        categories_labels = [
            cat.replace("Development (Character and Setting)", "Development")
            for cat in categories
        ]

        # Initialize bar data
        bar_width = 0.05  # Reduce bar width to make bars narrower
        gap_width = 0.05  # Gap between methods
        x = np.arange(len(categories)) * 0.65  # Position of categories

        # Plot setup
        fig, ax = plt.subplots(figsize=(7, 4))

        for i, method in enumerate(methods_to_compare):
            if method not in method_source_wise_results:
                continue

            # Get data for the source in the method
            if source not in method_source_wise_results[method]:
                continue

            method_data = method_source_wise_results[method][source]

            # Prepare bar segments
            expts = [method_data[cat].get("expts", 0) for cat in categories]
            ties = [method_data[cat].get("Tie", 0) for cat in categories]
            vanilla = [method_data[cat].get("vanilla", 0) for cat in categories]

            # Bottom positions for stacked bars
            bottom_tie = np.array(expts)
            bottom_vanilla = bottom_tie + np.array(ties)

            # Plot bars for the method with common colors
            method_x = x + i * (
                bar_width + gap_width
            )  # Add gap for differentiation between methods
            ax.bar(
                method_x,
                expts,
                bar_width,
                color=common_colors["expts"],
                label=f"{method_alias[method]} - expts" if i == 0 else "",
            )
            ax.bar(
                method_x,
                ties,
                bar_width,
                bottom=bottom_tie,
                color=common_colors["Tie"],
                label=f"{method_alias[method]} - Tie" if i == 0 else "",
            )
            ax.bar(
                method_x,
                vanilla,
                bar_width,
                bottom=bottom_vanilla,
                color=common_colors["vanilla"],
                label=f"{method_alias[method]} - vanilla" if i == 0 else "",
            )

            # Add method names directly below respective bars
            for pos in method_x:
                ax.text(
                    pos,
                    -0.05,
                    method_alias[method],
                    ha="center",
                    va="top",
                    fontsize=12,
                    rotation=15,
                    color="black",
                    transform=ax.get_xaxis_transform(),
                )

        # Adjust category labels to be tilted and flushed downward
        ax.set_xticks(x + (len(methods_to_compare) - 1) * (bar_width + gap_width) / 2)
        ax.set_xticklabels(
            categories_labels, rotation=0, ha="center", fontsize=14, y=-0.15
        )

        # Formatting
        ax.set_title(f"{source_alias[source]}", fontsize=16)
        # ax.set_xlabel('Category', fontsize=14)
        ax.set_ylabel("Win-Rate Proportion", fontsize=14)

        # Add a single legend
        handles = [
            plt.Line2D([0], [0], color=common_colors["expts"], lw=4, label="Method"),
            plt.Line2D([0], [0], color=common_colors["Tie"], lw=4, label="Tie"),
            plt.Line2D(
                [0], [0], color=common_colors["vanilla"], lw=4, label="Average Author"
            ),
        ]
        ax.legend(handles=handles, fontsize=14, loc="upper right")

        # Save the plot to disk
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{source.split('.')[0]}.png")
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Consolidate results from different sources for each method"
    )
    # model choice
    parser.add_argument(
        "--model_choice",
        type=int,
        default=1,
        help="Choice of the Model: 1. GPT-4o, 2. LLama-3.1-70B, 3. GPT-4o-mini, 4. Prometheus, 5. o4-mini",
    )
    # llama (store_true)
    parser.add_argument(
        "--llama", action="store_true", help="Consolidate results for llama 8B"
    )
    # llama (store_true)
    parser.add_argument(
        "--llama70", action="store_true", help="Consolidate results for llama 70B"
    )
    # faithfulness (store_true)
    parser.add_argument(
        "--faith", action="store_true", help="Author Sheet Score (Faithfulness)"
    )
    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()

    llama = args.llama
    faith = args.faith
    model_choice = args.model_choice

    if llama:
        llama_suffix = "_llama"
    elif args.llama70:
        llama_suffix = "_llama70"
    else:
        llama_suffix = ""

    if faith:
        root_dir = f"author_sheet_score_schema{llama_suffix}"
    else:
        root_dir = f"llm_evaluation_shuffle_score{llama_suffix}"

    # initialize the dictionary to store the results
    method_source_wise_results = defaultdict(dict)

    # iterate over directories in root_dir
    for method in os.listdir(root_dir):
        if "_old" in method:
            continue
        # considering only GPT evaluated data
        method_path = f"{root_dir}/{method}/{model_choice}"
        # iterate over sources in method_path
        for source in os.listdir(method_path):
            if 'old' in source or '_rerun' in source:
                continue

            source_path = f"{method_path}/{source}"
            # read source file
            with open(source_path, "r") as f:
                source_data = json.load(f)
            catwise_winners = get_catwise_winners(source_data, faith)
            # store the results
            method_source_wise_results[method][source] = catwise_winners

    # methodwise results (normalize across sources)
    consider_methods = [
        "oracle",
        "vanilla_few_shot",
        "delta",
        "schema",
        "schema_persona",
        "delta_schema",
        "delta_schema_persona",
    ]
    categorywise_method_results = defaultdict(dict)
    for method, source_data in method_source_wise_results.items():
        # if method not in consider_methods:
        #     continue
        catwise_winners = defaultdict(dict)
        for source, catwise_winners_source in source_data.items():
            for cat, win_dict in catwise_winners_source.items():
                for key, val in win_dict.items():
                    catwise_winners[cat][key] = catwise_winners[cat].get(key, 0) + val

        # normalize the results
        for cat, win_dict in catwise_winners.items():
            total = sum(win_dict.values())
            for key in win_dict:
                win_dict[key] /= total

        # sort catwise_winners based on values
        for cat, win_dict in catwise_winners.items():
            catwise_winners[cat] = {
                k: round(v, 2)
                for k, v in sorted(
                    win_dict.items(), key=lambda item: item[1], reverse=True
                )
            }
            if method in consider_methods:
                categorywise_method_results[cat][
                    method
                ] = f"{round(win_dict.get('expts', 0)*100, 2)} - ({round(win_dict.get('vanilla', 0)*100, 2)})"

        method_source_wise_results[method]["overall.json"] = catwise_winners

    # construct rows for table
    rows = []
    for cat, method_dict in categorywise_method_results.items():
        row = {"category": cat}
        for method in consider_methods:
            row[method] = method_dict.get(method, "NA")
        rows.append(row)

    # output dir
    output_dir = f"consolidate_results/{model_choice}/{root_dir}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # write the results to the output_dir
    for method, source_data in method_source_wise_results.items():
        for source, catwise_winners in source_data.items():
            output_sub_path = f"{output_dir}/{method}"
            if not os.path.exists(output_sub_path):
                os.makedirs(output_sub_path)
            # note source already has .json extension
            output_path = f"{output_sub_path}/{source}"
            with open(output_path, "w") as f:
                json.dump(catwise_winners, f, indent=4)

    # TODO: construct graph for each source
    create_graph(method_source_wise_results, output_dir)

    # write the table to a csv file
    df = pd.DataFrame(rows)
    df.to_csv(f"{output_dir}/catwise_winners{llama_suffix}.csv", index=False)

    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
