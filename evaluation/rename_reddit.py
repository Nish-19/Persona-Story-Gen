'''
Rename all files and directories
'''

import os

def rename_reddit_files_and_dirs(base_dir):
    """
    Rename all files named 'Reddit.json' to 'Reddit_old.json' and directories named 'Reddit' to 'Reddit_old'
    at depth 2 in the given directory.
    """
    for root, dirs, files in os.walk(base_dir):
        # Check if we are at depth 2
        if root[len(base_dir):].count(os.sep) == 2:
            # Rename files
            for file in files:
                if file == "Reddit.json":
                    old_path = os.path.join(root, file)
                    new_path = os.path.join(root, "Reddit_old.json")
                    os.rename(old_path, new_path)
                    print(f"Renamed file: {old_path} -> {new_path}")
            
            # Rename directories
            for dir_name in dirs:
                if dir_name == "Reddit":
                    old_dir_path = os.path.join(root, dir_name)
                    new_dir_path = os.path.join(root, "Reddit_old")
                    os.rename(old_dir_path, new_dir_path)
                    print(f"Renamed directory: {old_dir_path} -> {new_dir_path}")

# Specify the base directory
base_directory = "llm_evaluation_shuffle_score_stats"

# Call the function
rename_reddit_files_and_dirs(base_directory)