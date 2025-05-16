'''
Utility functions for finetuning and evaluating models.
'''

import os
import re
import sys
import json
import pandas as pd
from typing import Union, Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

MAX_LEN = 100_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_checkpoint_path(model_name: str):
    os.makedirs("saved_models", exist_ok=True)
    return f"saved_models/{model_name}"



def clear_evidence(writing_sheet):
    """
    Remove all 'Evidence' fields from the writing_sheet and clean up the text.
    Split the cleaned claims into a list of individual claims.
    """

    def sanitize_text(text):
        """
        Clean up hidden characters, excessive whitespaces, and normalize line endings.
        """
        # Normalize newlines to \n
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove non-breaking spaces and other invisible characters
        text = re.sub(r"[^\S\n]", " ", text)  # Replace non-space whitespace with space

        # Strip leading/trailing whitespaces and normalize multiple spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text
    
    def remove_evidence(user_sheet):
        """
        Remove evidence lines from the text.
        """

        # Match lines containing '- Evidence:' and remove them
        cleaned_sheet = re.sub(
            r" - Evidence:.*?(?=(\d+\.|$))", "", user_sheet, flags=re.DOTALL
        )

        # Replace multiple consecutive spaces or newlines with a single newline
        cleaned_sheet = re.sub(r"\s*\n\s*", "\n", cleaned_sheet.strip())

        # replace ** with ''
        cleaned_sheet = cleaned_sheet.replace("**", "")

        return cleaned_sheet
    
    # pre-defined categories
    categories = [
        "Plot",
        "Creativity",
        "Development (Character and Setting)",
        "Language Use",
    ]

    writing_sheet_categories = {}
    
    # extract elements
    for cctr, cat in enumerate(categories):
        # extract text between cat and categories[cctr+1]
        # find index of ### {cat}
        cat_idx = writing_sheet.find(f"### **{cat}**")
        # find index of ### {next category}
        if cctr == len(categories) - 1:
            next_cat_idx = len(writing_sheet)
        else:
            next_cat_idx = writing_sheet.find(
                f"### **{categories[cctr+1]}**"
            )
        # extract the text
        writing_sheet_temp = writing_sheet[
            cat_idx + len(f"### **{cat}**") : next_cat_idx
        ]

        # Sanitize extracted text
        writing_sheet_temp = sanitize_text(writing_sheet_temp)

        # Clear evidence from the writing sheet
        writing_sheet_categories[cat] = remove_evidence(
            writing_sheet_temp
        )

    final_sheet = ""
    # combine the categories into a single string
    for cat, text in writing_sheet_categories.items():
        final_sheet += f"### {cat}\n{text}\n"
    # remove the last newline
    final_sheet = final_sheet.rstrip("\n")

    return final_sheet
    

def extract_writing_sheet(sheet_output, key="combined_author_sheet"):
    """
    extract text between the tags <user_writing_sheet></user_writing_sheet>
    """
    sheet = re.search(rf"<{key}>(.*?)</{key}>", sheet_output, re.DOTALL).group(
        1
    )
    if not sheet:
        sheet = sheet_output
    return sheet


def load_data(split='profile', writing_sheet=False, writing_summary=False):
    '''
    returns a pandas dataframe with the data (source, writing prompt, story)
    '''

    if writing_sheet:
        writing_sheet_suffix = "_writing_sheet"
    elif writing_summary:
        writing_sheet_suffix = "_writing_summary"
    else:
        writing_sheet_suffix = ""

    # check if data already exists
    if os.path.exists(f'../datasets/finetune_data{writing_sheet_suffix}/{split}.csv'):
        # load the data
        finetune_df = pd.read_csv(f'../datasets/finetune_data{writing_sheet_suffix}/{split}.csv')
        if split == 'profile':
            val_finetune_df = pd.read_csv(f'../datasets/finetune_data{writing_sheet_suffix}/val.csv')
        else:
            val_finetune_df = None
        return finetune_df, val_finetune_df
    
    ignore_files = os.listdir("results/vanilla/Reddit_old")

    # define sources
    sources = ["Reddit", "AO3", "Storium", "narrativemagazine", "newyorker"]
    sources = [sources[0]] # only use reddit for now
    
    # store list
    finetune_data = []
    if split == 'profile':
        val_finetune_data = []

    # iterate over sources
    for source in sources:
        split_dir = f'../datasets/data_splits/data/{source}/{split}'
        story_dir = f'../datasets/{source}/selected_human_with_prompts/'
        if writing_sheet:
            writing_sheet_dir = f'user_profile/delta_schema/{source}/'
        if writing_summary:
            writing_sheet_dir = f'user_profile/schema/{source}/'

        # iterate over files in split_dir 
        for file in os.listdir(split_dir):
            # check if file is in ignore_files
            if file in ignore_files:
                continue
            if file.endswith('.json'):
                # load the split file
                with open(os.path.join(split_dir, file), 'r') as f:
                    data = json.load(f)
                # load the story file
                with open(os.path.join(story_dir, file), 'r') as f:
                    story_data = json.load(f)
                

                # load writing sheet file
                if writing_sheet or writing_summary:
                    with open(os.path.join(writing_sheet_dir, file), 'r') as f:
                        writing_sheet_data = json.load(f)
                    
                    if len(writing_sheet_data) == 1:
                        user_profile = extract_writing_sheet(
                            writing_sheet_data[-1], key="writing_style"
                        )
                    else:
                        user_profile = extract_writing_sheet(
                            writing_sheet_data[-1], key="combined_author_sheet"
                        )
                    
                    # clean up the user_profile
                    user_profile = clear_evidence(user_profile)
                    
                else:
                    user_profile = None

                # create dict of story data
                story_dict = {}
                for item in story_data:
                    story_dict[item['writing_prompt']] = item['comment']
                
                # set val size to 10% of the data
                val_size = int(len(data) * 0.2)
                # iterate over the items
                for ictr, item in enumerate(data):
                    # consider only first 5 items for the test set
                    if split == 'test' and ictr > 4:
                        break
                    wp = item['writing_prompt']
                    story = story_dict[wp]
                    # create dict of data
                    finetune_sample = {
                        'source': source,
                        'identifier': f"{file}_{ictr}",
                        'writing_prompt': wp,
                        'story': story
                    }
                    if writing_sheet or writing_summary:
                        finetune_sample['writing_sheet'] = user_profile
                    # check if split is profile and if ictr is greater than len(data) - val_size
                    if split == 'profile' and ictr > len(data) - val_size - 1:
                        # append to val_finetune_data
                        val_finetune_data.append(finetune_sample)
                    else:
                        # append to list
                        finetune_data.append(finetune_sample)
    
    # create dataframe
    finetune_df = pd.DataFrame(finetune_data)


    # save dataframe
    finetune_data_dir = f'../datasets/finetune_data{writing_sheet_suffix}'
    if not os.path.exists(finetune_data_dir):
        os.makedirs(finetune_data_dir)
    
    finetune_df.to_csv(f'{finetune_data_dir}/{split}.csv', index=False)

    if split == 'profile':
        # create dataframe for val data
        val_finetune_df = pd.DataFrame(val_finetune_data)
        val_finetune_df.to_csv(f'{finetune_data_dir}/val.csv', index=False)
    else:
        val_finetune_df = None

    # return dataframe
    return finetune_df, val_finetune_df


def get_base_model(base_model_name: str, quantize: bool) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = "<|finetune_right_pad_id|>" # NOTE: this is a special padding token for llama, it's important to not set this to eot or any regularly occurring token or will mess with trl code
    print(f"Loading model {'with' if quantize else 'without'} quanitization: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        pad_token_id=tokenizer.pad_token_id,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        ) if quantize else None,
        # f32 seems helpful for train/test time consistency when quantizing, bf16 performs best for non-quantized
        torch_dtype=torch.float32 if quantize else torch.bfloat16,
        device_map={"": 0}
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1
    return base_model, tokenizer

def get_model(model: PreTrainedModel, test: bool,
              model_name: Union[str, List[str]] = None, pt_model_name: str = None,
              r: int = None, lora_alpha: int = None,
              quantize: bool = False, use_gradient_checkpointing: bool = True) -> Union[PeftModel, PreTrainedModel]:
    if test and model_name:
        # Note we are loading adapter on quantized model and not merging
        # Recommended here - https://huggingface.co/docs/trl/main/en/dpo_trainer#downsides-to-merging-qlora-before-dpo-approach-2
        # Also prevents empty responses generated by Llama models
        model = PeftModel.from_pretrained(model, get_checkpoint_path(model_name))
    elif not test:
        if quantize:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
        if pt_model_name:
            print(f"Initializing trainable model from pre-trained LoRA adapters: {pt_model_name}")
            model = PeftModel.from_pretrained(model, get_checkpoint_path(pt_model_name), is_trainable=True, adapter_name="default")
            model.load_adapter(get_checkpoint_path(pt_model_name), is_trainable=False, adapter_name="lora_ref")
        else:
            print("Initializing trainable model with new LoRA adapters")
            peft_config = LoraConfig(
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                inference_mode=False,
            )
            model = get_peft_model(model, peft_config)
    else:
        print("Using inference-time model with pre-trained weights")
    return model

def get_prompt(sample, tokenizer: PreTrainedTokenizer = None, args = None):
    source = sample["source"]
    writing_prompt = sample["writing_prompt"]
    length = sample["story"].count(" ") + 1

    if not args.writing_sheet and not args.writing_summary:
        if args.test:
            # context = f"Source: {source}\tWriting Prompt: {writing_prompt}\tLength: {length} words"
            context = f"Writing Prompt: {writing_prompt}\tLength: {length} words"
            system_prompt = f"You are a story writer on Reddit's r/WritingPrompts platform tasked to write a story with the following Writing Prompt and Length (number of words)."
        else:
            context = f"Writing Prompt: {writing_prompt}"
            system_prompt = f"You are a story writer on Reddit's r/WritingPrompts platform tasked to write a story with the following Writing Prompt."
    else:
        # note include writing sheet in the prompt
        writing_sheet = sample["writing_sheet"]
        if args.test:
            context = f"Writing Sheet: {writing_sheet}\tWriting Prompt: {writing_prompt}\tLength: {length} words"
            system_prompt = f"You are a story writer on Reddit's r/WritingPrompts platform tasked to write a story with the following Writing Prompt, Length (number of words) and a Writing Sheet describing the writer's story-writing characterestics across four narrative categories - Plot, Creativity, Development, and Language Use."
        else:
            context = f"Writing Sheet: {writing_sheet}\tWriting Prompt: {writing_prompt}"
            system_prompt = f"You are a story writer on Reddit's r/WritingPrompts platform tasked to write a story with the following Writing Prompt and a Writing Sheet describing the writer's story-writing characterestics across four narrative categories - Plot, Creativity, Development, and Language Use."
    
    if tokenizer is not None:
        return tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ], tokenize=False, add_generation_prompt=True)
    else:
        prompt = context
        return prompt

class SFTExpandedDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: PreTrainedTokenizer, args):
        self.data = []
        excluded = 0
        for _, sample in data.iterrows():
            # rewrite prompt
            prompt = get_prompt(sample, tokenizer, args)
            if len(prompt + sample["story"]) < MAX_LEN:
                self.data.append({"source": sample["source"], "identifier": sample["identifier"], "wp": sample["writing_prompt"], "prompt": prompt, "label": sample["story"] + tokenizer.eos_token})
            else:
                excluded += 1
        print(f"Num turns: {len(self.data)} ({excluded} excluded)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class SFTExpandedCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        self.tokenizer.padding_side = "right"
        tokens = self.tokenizer(
            [sample["prompt"] + sample["label"] for sample in batch],
            return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        input_ids = tokens.input_ids
        attn_mask = tokens.attention_mask
        prompt_tokens = self.tokenizer(
            [sample["prompt"] for sample in batch],
            return_tensors="pt", padding=True, add_special_tokens=False)
        prompt_lens = prompt_tokens.attention_mask.sum(dim=1)
        labels = input_ids.clone()
        labels[attn_mask == 0] = -100
        label_mask = torch.arange(input_ids.shape[1]).repeat(input_ids.shape[0], 1) < prompt_lens.unsqueeze(1)
        labels[label_mask] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels
        }


class TestingCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        self.tokenizer.padding_side = "left"
        tokens = self.tokenizer([sample["prompt"] for sample in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        return {
            "input_ids": tokens.input_ids,
            "attention_mask": tokens.attention_mask,
            "meta_data": batch
        }


def test(args, test_df):
    # Load model
    base_model, tokenizer = get_base_model(args.base_model, args.quantize)
    model = get_model(base_model, True, model_name=args.model_name, quantize=args.quantize)

    collator = TestingCollator(tokenizer)
    test_dataset = SFTExpandedDataset(test_df, tokenizer, args)
    data_loader = DataLoader(test_dataset, args.test_batch_size, collate_fn=collator, shuffle=False)

    # Save results
    save_dir = f"finetune/{args.model_name}"
    os.makedirs(save_dir, exist_ok=True)

    # Generate tutor turns
    results = []
    for batch in tqdm(data_loader):
        outputs = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=args.max_gen_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
        )
        pred_stories = tokenizer.batch_decode(outputs[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        results.extend([
            {**sample, "pred_story": pred_story} for sample, pred_story in zip(batch["meta_data"], pred_stories)
        ])

        # delete the label from the results
        for result in results:
            if "label" in result:
                del result["label"]
    
        with open(f"{save_dir}/test_results.json", "w") as f:
            json.dump(results, f, indent=4)
    
    print(f"Test results saved to {save_dir}/test_results.json")