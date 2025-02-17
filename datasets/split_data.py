"""
split the dataset into 70 profile and 30 test per user
Outputs:
1. Directory containing number of profile and test data per user
2. Directory (same keys for all sources - date, writing prompt, story):
    - profile - contains profile data for each user (dictionary - list of dictionaries)
    - test - contains test data for each user (dictionary - list of dictionaries)
"""

import os
import json
from collections import defaultdict
from math import ceil
import pandas as pd
from datetime import datetime


def get_profile_test_data(data, profile_size, source):
    """
    Split the data into profile and test data
    """

    if source == "AO3":
        date_field = "published"
        date_format = "%d %b %Y"
        metadata_fields = ["title", "fandoms", "rating", "warnings", "relationships"]
    elif source == "narrativemagazine":
        date_field = "date"
        date_format = "%d %b %Y"
        metadata_fields = ["post_title"]
    elif source == "newyorker":
        date_field = "date"
        date_format = "%d %b %Y"
        metadata_fields = ["post_title"]
    elif source == "Reddit":
        date_field = "date_posted"
        date_format = "%Y-%m-%d %H:%M:%S"
        metadata_fields = []
    elif source == "Storium":
        date_field = "created_at"
        date_format = "%Y-%m-%d %H:%M:%S %Z"
        metadata_fields = ["story_name"]

    # sort the data based on date
    sorted_data = sorted(
        data, key=lambda x: datetime.strptime(x[date_field], date_format)
    )

    # split the data into profile and test data
    profile_data_raw = sorted_data[:profile_size]
    test_data_raw = sorted_data[profile_size:]

    # normalize the fields
    profile_data = list()
    for prof_data in profile_data_raw:
        # TODO: collect metadata
        metadata = dict()
        for field in metadata_fields:
            if "title" in field:
                metadata["story_name"] = prof_data[field]
            else:
                metadata[field] = prof_data[field]
        # add story length
        metadata["story_length"] = len(prof_data["comment"].split(" "))
        profile_data.append(
            {
                "date": prof_data[date_field],
                "metadata": metadata,
                "writing_prompt": prof_data["writing_prompt"],
                "story": prof_data["comment"],
            }
        )

    test_data = list()
    for t_data in test_data_raw:
        # TODO: collect metadata
        metadata = dict()
        for field in metadata_fields:
            if "title" in field:
                metadata["story_name"] = t_data[field]
            else:
                metadata[field] = t_data[field]
        # add story length
        metadata["story_length"] = len(t_data["comment"].split(" "))
        test_data.append(
            {
                "date": t_data[date_field],
                "metadata": metadata,
                "writing_prompt": t_data["writing_prompt"],
                "story": t_data["comment"],
            }
        )

    return profile_data, test_data


def main():
    data_sources = ["AO3", "narrativemagazine", "newyorker", "Reddit", "Storium"]
    data_split_per_source = defaultdict(dict)
    test_ratio = 0.3

    output_dir_numbers = "data_splits/stats"
    if not os.path.exists(output_dir_numbers):
        os.makedirs(output_dir_numbers)

    output_dir_data = "data_splits/data"
    if not os.path.exists(output_dir_data):
        os.makedirs(output_dir_data)

    # iterate through each data source
    for source in data_sources:
        data_dir = f"{source}/selected_human_with_prompts"
        # iterate through each file in the data source
        for file in os.listdir(data_dir):
            user = file.split(".")[0]
            file_path = os.path.join(data_dir, file)
            with open(file_path, "r") as f:
                data = json.load(f)
                # split the data into 70% profile and 30% test
                proposed_test_size = ceil(len(data) * test_ratio)
                test_size = proposed_test_size if proposed_test_size > 0 else 1
                profile_size = len(data) - test_size

                # TODO: sort data based on date return profile and test data
                profile_data, test_data = get_profile_test_data(
                    data, profile_size, source
                )

                if profile_size > 0:
                    # TODO: include raw indices corresponding to date sorted order
                    data_split_per_source[source][user] = {
                        "profile": profile_size,
                        "test": test_size,
                    }

                # save the profile and test data

                # output_path_profile_dir
                output_path_profile_dir = f"{output_dir_data}/{source}/profile"
                if not os.path.exists(output_path_profile_dir):
                    os.makedirs(output_path_profile_dir)

                # output_path_test_dir
                output_path_test_dir = f"{output_dir_data}/{source}/test"
                if not os.path.exists(output_path_test_dir):
                    os.makedirs(output_path_test_dir)

                # output_path_profile
                with open(f"{output_path_profile_dir}/{user}.json", "w") as f:
                    json.dump(profile_data, f, indent=4)

                # output_path_test
                with open(f"{output_path_test_dir}/{user}.json", "w") as f:
                    json.dump(test_data, f, indent=4)

        # save the data split
        output_path_numbers = os.path.join(output_dir_numbers, f"{source}.json")
        with open(output_path_numbers, "w") as f:
            json.dump(data_split_per_source[source], f, indent=4)

    # aggregate the data split across all sources
    data_split_stats = list()
    for source in data_sources:
        profile_size = 0
        test_size = 0
        for user, split in data_split_per_source[source].items():
            profile_size += split["profile"]
            test_size += split["test"]
        data_split_stats.append(
            {"Source": source, "Profile size": profile_size, "Test size": test_size}
        )

    # calculate the total data split for all sources

    total_profile_size = 0
    total_test_size = 0
    for row in data_split_stats:
        total_profile_size += row["Profile size"]
        total_test_size += row["Test size"]
    data_split_stats.append(
        {
            "Source": "Total",
            "Profile size": total_profile_size,
            "Test size": total_test_size,
        }
    )

    # save the data split stats
    df = pd.DataFrame(data_split_stats)
    df.to_csv(os.path.join(output_dir_numbers, "data_split_stats.csv"), index=False)


if __name__ == "__main__":
    main()
