import transformers
import torch

model_id = "/datasets/ai/llama3/meta-llama/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/33101ce6ccc08fa6249c10a543ebfcac65173393"

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)
print(pipeline.model.hf_device_map)
output =  pipeline("What is the capital of France?")
print(output)