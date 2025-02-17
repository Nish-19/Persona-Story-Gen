# Persona-Story-Gen

Code for the Personalized Story Generation Project

## Datasets

```cd datasets/data_splits/data```

```
├── AO3
├── narrativemagazine
├── newyorker
├── Reddit
└── Storium
```

Each source directory above contains two directories - profiling (```profile```) and the generation (```test```) sets. Both ```profile``` and ```test``` contain a list of files each corresponding to an author. Each author file contains a list of ```writing_prompt``` and corresponding ```story```.

Optional: If you want to re-generate the writing prompts navigate to ```cd generate_prompts```

## Experiments
```cd experiments```

```python methods.py --help``` for a details on all arguments. 

Use ```--source``` to specific name of data source (ex: Reddit), ```--llama``` and ```--llama70``` for LLama 3.1 8B and LLama 3.1 70B models respectively along with the commands given below.

Output directories
* Personalized Stories: ```results``` for GPT-4o, ```results_llama``` for LLama 3.1 8B, and ```results_llama70``` for LLama 3.1 70B
* Author Writing Sheet/ Summary: ```user_profile```
* Story Rules: ```story_rules```
* Persona descriptions: ```persona```

### Average Author

```python methods.py --choice 1```

The output is stored in directory ```vanilla```

### RAG

```python methods.py --choice 1 --few_shot```

The output is stored in directory ```vanilla_few_shot```

### Delta

1. Generate Average Author Stories for the profiling set 
```python methods.py --choice 1 --is_profile```

2. Generate rules for profiling set 
```python methods.py --extract_rules --is_profile```

3. Generate personalized stories using Delta
```python methods.py --choice 4```

The output is stored in directory ```delta```

### Writing Summary

```python methods.py --choice 3 --persona```

The output is stored in directory ```schema_persona```

### Writing Sheet 
```python methods.py --choice 5 --persona```

The output is stored in directory ```delta_schema_persona```

For nP variants of Writing Summary and Sheet (i.e., ablation without using persona description, do not include ```--persona``` in the command. The output is saved in ```schema``` and ```delta_schema``` respectively.)

## GPT-4o-as-judge Evaluation 

```cd evaluation```

### Faithfulness to Writing History 

Prompt GPT-4o for evaluation 
```python author_sheet_score_evaluation.py --source <source_name> --choice <choice_number>```

Compute win-rates
```python pool_author_sheet_score_evaluation.py --source <source_name> --choice <choice_number>```

### Similarity to Author Story 

Prompt GPT-4o for evaluation 
```python llm_evaluation_shuffle.py --source <source_name> --choice <choice_number>```

Compute win-rates
```python pool_llm_evaluation_score.py --source <source_name> --choice <choice_number>```

```<source_name>``` is the name of data source. ```<choice_number>``` is the choice of the method which can be found above in ```Experiments``` section. Additionally argument ```--persona``` should be used for Writing Sheet and Summary. ```--llama```, and ```--llama70``` for evaluating generations for LLama 3.1 8B and LLama 3.1 70B respectively. 

### Category-wise win-rates

```python consolidate_results.py```

```--faith``` for Faithfulness to Writing History. ```--llama```, and ```--llama70``` for LLama 3.1 8B and LLama 3.1 70B respectively. 

## Traditional Metrics 

```
cd traditional_evaluation
python get_scores.py --source <source_choice> 
python consolidate_results.py
```

For ```get_scores.py``` use the same arguments as mentioned Experiments for a particular method. Additionally use ```--compute_gt``` to get scores for the ground-truth author story. 