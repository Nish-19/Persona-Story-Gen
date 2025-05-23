# Persona-Story-Gen

## Datasets

Navigate to the datasets directory:

```
cd datasets/data_splits/data
```

The dataset is organized as follows:

```
├── AO3
├── narrativemagazine
├── newyorker
├── Reddit
└── Storium
```

Each source directory contains two subdirectories: `profile` and `test`, corresponding to the **profiling** and **generation** sets, respectively. 

Each directory contains a list of files, with each file corresponding to an author. Each author file consists of:
- A list of `writing_prompt`
- The corresponding `url`, which can be used to download the story.

For **Storium**, you need to request access to the dataset from the original authors and collect the `game_pid` story. You can find more details at [Storium Dataset](https://storium.cs.umass.edu/).

### Optional: Re-generate Writing Prompts

To regenerate writing prompts, follow instructions in the `generate_prompts` directory:

```
cd generate_prompts
```

---

## Experiments

Navigate to the experiments directory:

```
cd experiments
```

Run the following command to see details of available arguments:

```
python methods.py --help
```

Use the `--source` argument to specify a data source (e.g., `Reddit`). 

### Model Selection
- Default GPT-4o
- `--llama` for LLaMA 3.1 8B
- `--llama70` for LLaMA 3.1 70B

Output directories:
- **Personalized Stories**: 
  - `results` (GPT-4o)
  - `results_llama` (LLaMA 3.1 8B)
  - `results_llama70` (LLaMA 3.1 70B)
- **Author Sheet/Summary**: `user_profile`
- **Story Rules**: `story_rules`
- **Persona Descriptions**: `persona`

### Average Author

```
python methods.py --choice 1
```

Output directory: `vanilla`

### RAG

```
python methods.py --choice 1 --few_shot
```

Output directory: `vanilla_few_shot`

### Delta

1. Generate Average Author Stories for the profiling set:

   ```
   python methods.py --choice 1 --is_profile
   ```

2. Generate rules for the profiling set:

   ```
   python methods.py --extract_rules --is_profile
   ```

3. Generate personalized stories using Delta:

   ```
   python methods.py --choice 4
   ```

Output directory: `delta`

### Summary

```
python methods.py --choice 3 --persona
```

Output directory: `schema_persona`

### Sheet

```
python methods.py --choice 5 --persona
```

Output directory: `delta_schema_persona`

#### nP Variants (Ablation without Persona Descriptions)
For the nP variants of Summary and Sheet (i.e., ablation without using persona descriptions), **omit** the `--persona` flag.

- Summary nP output: `schema`
- Sheet nP output: `delta_schema`

---

## LLM-as-Judge Evaluation

Navigate to the evaluation directory:

```
cd evaluation
```

### Faithfulness to Writing History

1. Prompt LLM for evaluation:

   ```
   python author_sheet_score_evaluation.py --model_choice 5 --source <source_name> --choice <choice_number>
   ```

2. Compute win-rates:

   ```
   python pool_author_sheet_score_evaluation.py --model_choice 5 --source <source_name> --choice <choice_number>
   ```

```--model_choice 5``` for using OpenAI o4-mini for evaluation.

### Similarity to Author Story

1. Prompt LLM for evaluation:

   ```
   python llm_evaluation_shuffle.py --model_choice 5 --source <source_name> --choice <choice_number>
   ```

2. Compute win-rates:

   ```
   python pool_llm_evaluation_score.py --model_choice 5 --source <source_name> --choice <choice_number>
   ```

#### Notes:
- `<source_name>` refers to the dataset source (e.g., `Reddit`).
- `<choice_number>` corresponds to the method choice (see **Experiments** section).
- Use the `--persona` argument for Sheet and Summary evaluations.
- Use `--llama` and `--llama70` for evaluating generations from LLaMA 3.1 8B and LLaMA 3.1 70B, respectively.

### Category-wise Win-Rates

```
python consolidate_results.py
```

- Use `--faith` for Faithfulness to Writing History.
- Use `--llama` and `--llama70` for LLaMA 3.1 8B and LLaMA 3.1 70B.

---

## Traditional Metrics Evaluation

Navigate to the traditional evaluation directory:

```
cd traditional_evaluation
```

1. Compute evaluation scores:

   ```
   python get_scores.py --source <source_choice>
   ```

2. Consolidate results:

   ```
   python consolidate_results.py
   ```

#### Notes:
- Use the same arguments as in **Experiments** for selecting a specific method.
- Add `--compute_gt` to compute scores for the **ground-truth** author story.

---

## 📜 Citation
```
@article{kumar2025whose,
  title={Whose story is it? Personalizing story generation by inferring author styles},
  author={Kumar, Nischal Ashok and Pham, Chau Minh and Iyyer, Mohit and Lan, Andrew},
  journal={arXiv preprint arXiv:2502.13028},
  year={2025}
}
```

---