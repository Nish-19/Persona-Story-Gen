'''
extract batch output from OpenAI
'''

import os 
import json 
from collections import defaultdict

def process_custom_id(custom_id, source_names):
    '''
    split custom id into constituent parts
    '''
    custom_id_parts = custom_id.split('_')
    source = custom_id_parts[0]
    if source not in source_names:
        source = ''
    order = custom_id_parts[-1]
    category = custom_id_parts[-2]
    if source not in source_names:
        run_name = custom_id.split(f'_{category}_{order}')[0]
    else:
        run_name = '_'.join(custom_id_parts[1:-2])

    return source, category, order, run_name

def main():
    expt_name = 'delta_schema_persona.jsonl'
    batch_file_path = f'batch_openai_output/{expt_name}'
    with open(batch_file_path, 'r') as f:
        results_list = f.readlines()
    
    root_path = f'llm_evaluation_shuffle_score_llama70/{expt_name.split(".")[0]}/1'
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    
    source_wise_results = defaultdict(dict)
    sizes = {'Reddit': 228, 'AO3': 548, 'Storium': 708, 'narrativemagazine': 764, 'newyorker': 824}

    # iterate through each result
    for rctr, result_raw in enumerate(results_list):
        result_data = json.loads(result_raw)

        # custom id
        custom_id = result_data['custom_id']
        output = result_data["response"]["body"]["choices"][0]["message"]["content"]
        source, category, order, run_name = process_custom_id(custom_id, list(sizes.keys()))
        if source not in sizes:
            # TODO: assign source based on rctr 
            for psource, size in sizes.items():
                if rctr < size:
                    source = psource
                    break

        # add results to source wise dict
        if run_name not in source_wise_results[source]:
            source_wise_results[source][run_name] = {}
        if len(source_wise_results[source][run_name]) == 0:
            source_wise_results[source][run_name] = {category:{"1":output, 2:f"A: {order}"}}
        else:
            source_wise_results[source][run_name][category] = {"1":output, 2:f"A: {order}"}
    
    # dump results source-wise
    for source, source_results in source_wise_results.items():
        source_path = f'{root_path}/{source}.json'
        with open(source_path, 'w') as f:
            json.dump(source_results, f, indent=4)

if __name__ == "__main__":
    main()