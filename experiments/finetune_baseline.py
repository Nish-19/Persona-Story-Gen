'''
Code for finetuning a baseline model on a dataset.
'''

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import pandas as pd
from transformers import PreTrainedTokenizer, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset

def main():
    pass

if __name__ == "__main__":
    main()