import datasets
import os
import time
from nomic import AtlasProject
from tags import tag_list
import json


dataset = datasets.load_dataset("../datasets/OpenOrca", data_files=['1M-GPT4-Augmented.parquet'], split="train")
dataset_size = 1000000
print(dataset)
df = dataset.to_pandas()
#removes any duplicate ids or questions
df.drop_duplicates(subset=['id'], inplace=True, keep='first')
df.drop_duplicates(subset=['question'], inplace=True, keep='first')
dataset = datasets.Dataset.from_pandas(df)

print(dataset)

def BatchTagFunction(examples):
    batch_size = len(examples['response'])
    keep = [True] * batch_size

    for i in range(batch_size):
        example = {
            'id': examples['id'][i],
            'response': examples['response'][i],
            'system_prompt': examples['system_prompt'][i],
            'question': examples['question'][i],
        }
        for tag in tag_list:
            if tag.evaluate(example):
                keep[i] = False
                break

    return keep

cpu_cores = os.cpu_count() or 1

dataset = dataset.filter(BatchTagFunction, num_proc=cpu_cores, batch_size=1000, batched=True)
print(dataset)

dataset.to_parquet("../datasets/OpenOrca/OpenOrca_GPT4_820K_Filtere1.parquet")