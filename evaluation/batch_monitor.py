from openai import OpenAI
import json
import os

def main():
    # Read API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    print('Using OpenAI API Key: ', api_key)
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # list of batches
    print(client.batches.list(limit=15))

    # retrieve a batch
    batch_job = client.batches.retrieve("batch_6788887e866c81909871487a7111b748")
    print(batch_job)
    # Get completed file
    results_list = []
    result_file_id = batch_job.output_file_id
    result = client.files.content(result_file_id).content
    result = result.decode('utf-8')
    result_entries = result.strip().split("\n")
    for r in result_entries:
        results_list.append(json.loads(r))
    
    # batch job input file name (metadata)
    metadata_file_name = batch_job.metadata['description'].split('/')[1]

    # output dir
    output_dir = 'batch_openai_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = f"{output_dir}/{metadata_file_name}"
    with open(output_file, 'w') as f:
        for result in results_list:
            f.write(json.dumps(result) + '\n')
    
    print('Length of results_list:', len(results_list))
  

if __name__ == "__main__":
    main()