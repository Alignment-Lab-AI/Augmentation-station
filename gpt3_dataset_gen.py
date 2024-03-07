import datasets
import os
import time
from nomic import AtlasProject
from tags import tag_list
import json

question_dataset = dataset = datasets.load_dataset("../datasets/OpenOrca", data_files=['OpenOrca_GPT4_820K_Filtered.parquet'], split="train")
questions = set(question_dataset['question'])

dataset = datasets.load_dataset("../datasets/OpenOrca", data_files=['3_5M-GPT3_5-Augmented.parquet'], split="train")
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
        if example['question'] in questions:
            keep[i] = False
            continue

        for tag in tag_list:
            if tag.evaluate(example):
                keep[i] = False
                break

    return keep

cpu_cores = os.cpu_count() or 1

dataset = dataset.filter(BatchTagFunction, num_proc=cpu_cores, batch_size=1000, batched=True)
print(dataset)

dataset.to_parquet("../datasets/OpenOrca/OpenOrca_GPT3_3_5M_Filtered.parquet")