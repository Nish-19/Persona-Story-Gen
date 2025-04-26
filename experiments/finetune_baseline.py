'''
Code for finetuning a baseline model on a dataset.
'''

import os
import json
import argparse
import wandb


import pandas as pd
from transformers import PreTrainedTokenizer, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset

from finetune_utils import load_data, get_prompt, SFTExpandedDataset, SFTExpandedCollator, get_checkpoint_path, get_base_model, get_model, test

MAX_LEN = 20_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_training_args(args):
    return TrainingArguments(
        output_dir=get_checkpoint_path(args.model_name),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.wd,
        max_grad_norm=args.gc or None,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        per_device_eval_batch_size=args.test_batch_size,
        eval_accumulation_steps=4,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to="wandb" if args.wandb else "none",
        label_names=["labels"],
    )

def parse_args():
    parser = argparse.ArgumentParser(description='Finetune a baseline model on a dataset.')
    # wandb
    parser.add_argument("--wandb", action="store_true", help="Log to wandb")
    # Modeling
    parser.add_argument("--model_choice", type=int, default=8, help="model size in B") # "meta-llama/Meta-Llama-3.1-8B-Instruct" "meta-llama/Llama-3.2-3B-Instruct"
    parser.add_argument("--base_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct") # "meta-llama/Meta-Llama-3.1-8B-Instruct" "meta-llama/Llama-3.2-3B-Instruct"
    parser.add_argument("--model_name")
    parser.add_argument("--pt_model_name")
    parser.add_argument("--quantize", action="store_true")
    # Training/Testing
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size at train-time")
    parser.add_argument("--test_batch_size", type=int, default=8, help="Batch size at test-time")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--gc", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Steps to accumulate gradients for")
    parser.add_argument("--r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max_gen_tokens", type=int, default=1500, help="Maximum number of tokens to generate")
    return parser.parse_args()

def main():
    # load arguments
    args = parse_args()
    # set test to False
    args.test = False

    if args.wandb:
        wandb.init(
            project="Persona Story Gen",
            name=f"{args.model_name}-run",
            config=args,
            tags=["llama", "peft", "finetune"]
        )

    if args.model_choice == 8:
        args.base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        args.grad_accum_steps = 64
    elif args.model_choice == 3:
        args.base_model = "meta-llama/Llama-3.2-3B-Instruct"
        args.grad_accum_steps = 32
    else:
        raise ValueError("Invalid model choice. Choose 8 or 3.")

    # load the dataset
    profile_df, val_df = load_data(split='profile')
    test_df, _ = load_data(split='test')

    # load model
    base_model, tokenizer = get_base_model(args.base_model, args.quantize)
    model = get_model(base_model, False, pt_model_name=args.pt_model_name, r=args.r, lora_alpha=args.lora_alpha, quantize=args.quantize)

    # create dataset and collator
    train_dataset = SFTExpandedDataset(profile_df, tokenizer, args)
    val_dataset = SFTExpandedDataset(val_df, tokenizer, args)
    collator = SFTExpandedCollator(tokenizer)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Train
    training_args = get_training_args(args)
    # training_args.world_size = 1

    print("### Training ###")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model()

    # Test
    # set test to True
    print("### Testing ###")
    args.test = True
    test(args, test_df)

if __name__ == "__main__":
    main()