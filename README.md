# Augmentation-station

This repository contains a collection of Python scripts for processing and augmenting data using the OpenAI API. The scripts are designed to work with JSONL files and perform various tasks such as indexing, templating, and filtering data.

## Scripts

### 1. `Generic_Augmentations.py`

This script processes a JSONL file asynchronously using the OpenAI API to generate augmentations based on templates. It performs the following tasks:

- Loads processed indices from a file to avoid reprocessing data.
- Indexes the input dataset if not already indexed.
- Loads templates from the `templates` directory.
- Constructs prompts by substituting placeholders in the templates with corresponding values from the dataset.
- Sends prompts to the OpenAI API asynchronously in batches.
- Handles API responses, including successful augmentations, rate limiting, and errors.
- Saves processed indices and handles rejections by writing them to separate files.

### 2. `gpt3_dataset_gen.py`

This script generates a filtered dataset using the GPT-3 model. It performs the following tasks:

- Loads the OpenOrca dataset from parquet files.
- Removes duplicate entries based on the 'id' and 'question' columns.
- Filters the dataset using a custom `BatchTagFunction` that applies a set of tags defined in the `tags.py` file.
- Saves the filtered dataset to a new parquet file.

### 3. `GPT4_dataset_gen.py`

This script generates a filtered dataset using the GPT-4 model. It follows a similar process to `gpt3_dataset_gen.py`:

- Loads the OpenOrca dataset from parquet files.
- Removes duplicate entries based on the 'id' and 'question' columns.
- Filters the dataset using a custom `BatchTagFunction` that applies a set of tags defined in the `tags.py` file.
- Saves the filtered dataset to a new parquet file.

### 4. `GPT4_prompts_to_run.py`

This script prepares a dataset of prompts to be run using the GPT-4 model. It performs the following tasks:

- Loads the filtered GPT-3 dataset from a parquet file.
- Loads prompts from a `prompts.json` file.
- Filters the dataset based on the system prompts specified in the loaded prompts.
- Checks if each dataset chunk has enough examples to meet the required count.
- Merges the filtered dataset chunks and shuffles the resulting dataset.
- Saves the merged dataset to a new parquet file.

### 5. `tags.py`

This file defines a set of tags and techniques used for filtering the datasets. Each technique is represented by a `Technique` class that has a name, description, and an evaluation function. The evaluation function takes an example object and returns a boolean value indicating whether the example should be filtered out or not.

The file also defines a list of refusal strings used to identify examples that should be filtered out based on the presence of these strings in the response.

## Usage

To use these scripts, follow these steps:

1. Set up the necessary environment variables, such as the OpenAI API base URL and any required API keys.

2. Prepare your input datasets in the appropriate formats (JSONL or parquet) and place them in the `data` directory.

3. Create a `templates` directory and add your template files (`.txt`) for the `Generic_Augmentations.py` script.

4. Run the desired script using Python, e.g., `python Generic_Augmentations.py`.

5. The processed datasets will be saved in the `data/output` directory, and any rejected or malformed entries will be saved in the `data/rejected` and `data/malformed` directories, respectively.

## Conclusion

These scripts provide a streamlined workflow for processing and augmenting datasets using the OpenAI API. They handle various tasks such as indexing, templating, filtering, and error handling. The `tags.py` file allows for customizable filtering techniques based on specific criteria.

By using these scripts, you can efficiently process large datasets and generate augmented versions using the GPT-3 and GPT-4 models. The filtered datasets can be used for further analysis or training purposes.
