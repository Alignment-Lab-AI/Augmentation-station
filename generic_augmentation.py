import os
import json
import hashlib
import asyncio
import aiohttp

BATCH_SIZE = 256
MAX_RETRIES = 99999999
RETRY_DELAY = 30
TOTAL_GENERATIONS = 100000 # Set your desired total number of generations here

if not os.path.exists('data/output'):
    os.makedirs('data/output')
if not os.path.exists('data/malformed'):
    os.makedirs('data/malformed')
if not os.path.exists('data/rejected'):
    os.makedirs('data/rejected')
if not os.path.exists('templates'):
    os.makedirs('templates')

class OpenAI_API:
    def __init__(self):
        print("Initializing...")
        self.session = None
        self.processed_indices = self.load_processed_indices()
        self.total_generations_processed = 0

    def load_processed_indices(self):
        print("Loading processed indices...")
        if os.path.exists('processed_indices.txt'):
            with open('processed_indices.txt', 'r') as f:
                indices = set(map(int, f.readlines()))
                print(f"Loaded {len(indices)} processed indices.")
                return indices
        print("No processed indices found.")
        return set()

    def save_processed_index(self, idx):
        print(f"Saving index {idx} as processed...")
        with open('processed_indices.txt', 'a') as f:
            f.write(str(idx) + '\n')

    def index_dataset(self, input_file_path):
        print("Checking if dataset needs indexing...")
        with open(input_file_path, 'r') as f:
            first_line = f.readline()
            if '"idx":' in first_line:
                print("Dataset is already indexed.")
                return

        print("Indexing dataset...")
        with open(input_file_path, 'r') as f:
            data = [json.loads(line) for line in f]

        for idx, item in enumerate(data):
            item['idx'] = idx

        with open(input_file_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print("Dataset indexed successfully.")

    def load_templates(self):
        print("Loading templates...")
        templates = []
        for filename in sorted(os.listdir('templates')):
            if filename.endswith('.txt'):
                with open(os.path.join('templates', filename), 'r') as f:
                    templates.append(f.read().strip())
        return templates

    def construct_prompt(self, data, template):
        """
        Constructs a prompt by substituting placeholders in the template with
        corresponding values from the data dictionary.

        :param data: A dictionary containing key-value pairs from the dataset.
        :param template: A string template with placeholders matching keys in the data dictionary.
        :return: A string with placeholders in the template replaced with data values.
        """
        return template.format_map(data)

    async def process_jsonl_file_async(self):
        templates = self.load_templates()
        template_idx = 0

        input_file_name = [f for f in os.listdir('data') if f.endswith('.jsonl')][0]
        input_file_path = os.path.join('data', input_file_name)
        self.index_dataset(input_file_path)

        output_file_name = input_file_name.replace('.jsonl', '_augmented.jsonl')
        output_file_path = os.path.join('data/output', output_file_name)

        async with aiohttp.ClientSession() as self.session:
            with open(input_file_path, 'r') as file:
                while self.total_generations_processed < TOTAL_GENERATIONS:
                    tasks = []
                    for _ in range(BATCH_SIZE):
                        line = file.readline()
                        if not line:
                            break

                        prompt_data = json.loads(line.strip())
                        if prompt_data['idx'] in self.processed_indices:
                            continue

                        original_template = prompt_data.get('template', '')
                        prompt_data['prompt'] = self.construct_prompt(prompt_data, templates[template_idx])
                        template_idx = (template_idx + 1) % len(templates)

                        task = asyncio.create_task(self.send_prompt(prompt_data['idx'], prompt_data, output_file_path))
                        tasks.append(task)

                        prompt_data['template'] = original_template

                    await asyncio.gather(*tasks)

        print("Processing complete.")


    async def send_prompt(self, idx, original_data, output_file_path):
        print(f"Sending prompt for idx {idx}...")
        base_url = os.environ['OPENAI_API_BASE']
        url = f"{base_url}/chat/completions"
        headers = {'Content-Type': 'application/json'}
        data = {
            "model": "mix",
            "max_tokens": 16000,
            "min_p": 0.2,
            "TFS": 0.75,
            "repetition_penalty": 1.2,
            "temperature": 1.3,
            "messages": [
                {
                    "role": "user",
                    "content": original_data['prompt']
                }
            ]
        }

        for attempt in range(MAX_RETRIES):
            try:
                async with self.session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"Received successful response for idx {idx}.")

                        content = result["choices"][0]["message"]["content"]

                        try:
                            unescaped_content = bytes(content, 'utf-8').decode('unicode_escape')
                            parsed_augmentation = json.loads(unescaped_content)

                            output_data = original_data.copy()
                            output_data['augmentation'] = parsed_augmentation

                            with open(output_file_path, "a") as outfile:
                                outfile.write(json.dumps(output_data) + '\n')
                            self.total_generations_processed += 1
                            self.save_processed_index(idx)
                            return output_data
                        except json.JSONDecodeError as e:
                            print(f"JSON Decode Error for idx {idx}. Error: {e}. Raw response content: {content}")
                            continue

                    elif response.status == 429:
                        print(f"Rate limit hit for idx {idx}. Retrying after {RETRY_DELAY} seconds...")
                        await asyncio.sleep(RETRY_DELAY)
                        continue
                    else:
                        error_content = await response.text()
                        print(f"Failed to get response for idx {idx}, status code: {response.status}, error: {error_content}")
                        if response.status == 403:
                            self.handle_rejection(original_data, error_content)
                            return original_data, None
                        await asyncio.sleep(RETRY_DELAY)

            except Exception as e:
                print(f"General Exception encountered for idx {idx}. Type: {type(e).__name__}, Error: {e}")
                await asyncio.sleep(RETRY_DELAY)

        print(f"Max retries reached for idx {idx}. Writing to rejected.")
        self.handle_rejection(original_data, "Max retries reached")
        return original_data, None

    def handle_rejection(self, data, error):
        print(f"Handling rejection for idx {data['idx']}...")
        hash_hex = hashlib.sha1(os.urandom(10)).hexdigest()
        rejected_file_path = os.path.join('data/rejected', f'{hash_hex}.jsonl')
        with open(rejected_file_path, "w") as f:
            f.write(json.dumps({'error': error, 'data': data}) + '\n')


if __name__ == "__main__":
    api = OpenAI_API()
    asyncio.run(api.process_jsonl_file_async())
