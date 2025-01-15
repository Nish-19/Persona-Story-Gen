# import transformers
# import torch

# model_id = "/datasets/ai/llama3/meta-llama/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/33101ce6ccc08fa6249c10a543ebfcac65173393"

# pipeline = transformers.pipeline(
#     "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
# )
# print(pipeline.model.hf_device_map)
# output =  pipeline("What is the capital of France?")
# print(output)

import time
from prompt_llm_utils import prompt_llama

start_time = time.time()

content1 = 'Write a story of 1000 words about an alien spicies invading the world'
content2 = 'Write a story of 1000 words about a mother who is a detective'
content3 = 'Write a story of 1000 words about an elf who is a wizard'

messages=[
    {"role": "user", "content": content3},
]

response = prompt_llama(messages, max_tokens=2000, temperature=0.0, top_p=1.0)

end_time = time.time()
total_time = end_time - start_time

print(response)
print(f"Total time taken: {total_time} seconds")