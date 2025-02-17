"""
get results table for the paper
"""

import os
import argparse
import re
from collections import defaultdict
import json
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Consolidate results from different sources for each method"
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
    # model choice
    parser.add_argument(
        "--model_choice",
        type=int,
        default=1,
        help="Choice of the Model: 1. GPT-4o, 2. LLama-3.1-70B",
    )
    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()

    llama = args.llama
    llama70 = args.llama70
    faith = args.faith
    model_choice = str(args.model_choice)

    print("Model Choice:", model_choice)
    print("Faithfulness:", faith)

    if llama:
        llama_suffix = "_llama"
    elif llama70:
        llama_suffix = "_llama70"
    else:
        llama_suffix = ""

    if llama or llama70:
        print("LLama Suffix:", llama_suffix.split("_")[1])

    if faith:
        root_dir = f"author_sheet_score_schema{llama_suffix}_stats"
    else:
        root_dir = f"llm_evaluation_shuffle_score{llama_suffix}_stats"

    methods = [
        "oracle",
        "vanilla_few_shot",
        "delta",
        "delta_schema_persona",
        "schema_persona",
        "delta_schema",
        "schema",
    ]
    methods_alias = [
        "Oracle",
        "RAG",
        "Delta",
        "Writing Sheet",
        "Writing Summary",
        "Writing Sheet nP",
        "Writing Summary nP",
    ]
    sources = ["AO3", "Reddit", "Storium", "narrativemagazine", "newyorker"]
    sources_alias = ["AO3", "Reddit", "Storium", "N.Magazine", "New Yorker"]

    method_source_wise_results = defaultdict(dict)
    for mctr, method in enumerate(methods):
        method_dir = f"{root_dir}/{method}/{model_choice}"
        all_expts, all_vanilla = [], []
        for sctr, source in enumerate(sources):
            source_dir = f"{method_dir}/{source}"
            stats_file = f"{source_dir}/winner_stats_score.json"
            with open(stats_file, "r") as f:
                results = json.load(f)
                expts_value, vanilla_value = results.get("expts", 0), results.get(
                    "vanilla", 0
                )
                expts_percent = round(expts_value / (sum(list(results.values()))) * 100)
                vanilla_percent = round(
                    vanilla_value / (sum(list(results.values()))) * 100
                )
                method_name, source_name = methods_alias[mctr], sources_alias[sctr]
                method_source_wise_results[method_name][
                    source_name
                ] = f"{expts_percent}-{vanilla_percent}"
                all_expts.append(expts_percent)
                all_vanilla.append(vanilla_percent)
        # calculate average for each method
        avg_expts = round(sum(all_expts) / len(all_expts))
        avg_vanilla = round(sum(all_vanilla) / len(all_vanilla))
        method_source_wise_results[method_name][
            "Overall"
        ] = f"{avg_expts}-{avg_vanilla}"

    # create a dataframe
    df = pd.DataFrame(method_source_wise_results)
    # output directory
    output_root_dir = "results_table"
    filename = f"{root_dir}_{model_choice}"
    os.makedirs(output_root_dir, exist_ok=True)
    # save the dataframe
    df.to_csv(os.path.join(output_root_dir, f"{filename}.csv"))


if __name__ == "__main__":
    main()
