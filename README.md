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


## Traditional Metrics 

## Human Evaluation 