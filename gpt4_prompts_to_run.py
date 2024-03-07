import datasets
import os
import time
from nomic import AtlasProject
from tags import tag_list
import json
#gpt3_dataset = datasets.load_dataset("../datasets/OpenOrca", data_files=['3_5M-GPT3_5-Augmented.parquet'], split="train")
#ids = set(gpt3_dataset['id'])

dataset = datasets.load_dataset("../datasets/OpenOrca", data_files=['OpenOrca_GPT3_3_5M_Filtered.parquet'], split="train")
dataset_size = 1000000

# Load dataset_split.json
with open('prompts.json') as f:
    prompts = json.load(f)['chuncks']

dataset_chuncks = []

for prompt in prompts:
    #print(f"Running {prompt['id']} {prompt['index']}")
    chunck_size = prompt['missing_count']
    # Filter dataset
    dataset_chunck = dataset.filter(lambda example: example['system_prompt'] == prompt['system_prompt'], num_proc=12)
    # Checks to make sure the dataset chunk is large enough
    if len(dataset_chunck) < chunck_size:
        #print(f"Skipping {prompt['id']} {prompt['index']}, not enough examples")
        #print the number of examples that are missing
        print(f"{prompt['id']} {prompt['index']}: Missing {chunck_size - len(dataset_chunck)} examples")
        continue
    # Split dataset and add to list

    dataset_chuncks.append(dataset_chunck.select(range(prompt['missing_count']))
    )

# Merge all of the datasets together
merged_dataset = datasets.concatenate_datasets(dataset_chuncks)
merged_dataset = merged_dataset.remove_columns(['response'])
merged_dataset = merged_dataset.add_column('response', [''] * len(merged_dataset))
merged_dataset = merged_dataset.shuffle()

# Only save if the dataset is larger than the target size
merged_dataset.to_parquet("../datasets/OpenOrca/GPT4_Prompts_To_Run.parquet")
print(merged_dataset)