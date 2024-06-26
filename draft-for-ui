import os
import json
import asyncio
import aiohttp
import argparse
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class OpenAI_API:
    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.processed_indices: set = self._load_processed_indices()
        self.total_generations_processed: int = 0
        self.templates: Dict[str, str] = {}
        self._ensure_directories()

    @staticmethod
    def _ensure_directories():
        os.makedirs('data/output', exist_ok=True)
        os.makedirs('templates', exist_ok=True)

    def _load_processed_indices(self) -> set:
        if os.path.exists('processed_indices.txt'):
            with open('processed_indices.txt', 'r') as f:
                return set(map(int, f.readlines()))
        return set()

    def _save_processed_index(self, idx: int):
        with open('processed_indices.txt', 'a') as f:
            f.write(f"{idx}\n")

    def _load_templates(self):
        for filename in os.listdir('templates'):
            if filename.endswith('.txt'):
                with open(os.path.join('templates', filename), 'r') as f:
                    self.templates[filename] = f.read().strip()

    def _construct_prompt(self, data: Dict, template: str) -> str:
        return template.format_map(data)

    async def process_jsonl_file(self):
        self._load_templates()
        template_list = list(self.templates.values())
        template_idx = 0

        input_file = self.config['input_file']
        input_path = os.path.join('data', input_file)
        output_path = os.path.join('data/output', input_file.replace('.jsonl', '_augmented.jsonl'))

        self._index_dataset(input_path)

        async with aiohttp.ClientSession() as self.session:
            with open(input_path, 'r') as infile, open(output_path, 'a') as outfile:
                while self.total_generations_processed < self.config['total_generations']:
                    batch = []
                    for _ in range(self.config['batch_size']):
                        line = infile.readline()
                        if not line:
                            break

                        data = json.loads(line.strip())
                        if data['idx'] in self.processed_indices:
                            continue

                        template = template_list[template_idx]
                        template_idx = (template_idx + 1) % len(template_list)
                        prompt = self._construct_prompt(data, template)

                        batch.append((data['idx'], data, template, prompt))

                    if not batch:
                        break

                    results = await asyncio.gather(*[self._send_prompt(*args) for args in batch])
                    for result in results:
                        if result:
                            json.dump(result, outfile)
                            outfile.write('\n')
                            outfile.flush()

        print(f"Processing complete. Generated {self.total_generations_processed} augmentations.")

    def _index_dataset(self, file_path: str):
        with open(file_path, 'r+') as f:
            first_line = f.readline()
            if '"idx":' not in first_line:
                print("Indexing dataset...")
                f.seek(0)
                data = [json.loads(line) for line in f]
                for idx, item in enumerate(data):
                    item['idx'] = idx
                f.seek(0)
                for item in data:
                    f.write(json.dumps(item) + '\n')
                f.truncate()
                print("Dataset indexed successfully.")
            else:
                print("Dataset is already indexed.")

    async def _send_prompt(self, idx: int, original_data: Dict, template: str, prompt: str) -> Optional[Dict]:
        url = f"{self.config['api_base']}/chat/completions"
        headers = {'Content-Type': 'application/json'}
        data = {
            "model": self.config['model'],
            "max_tokens": self.config['max_tokens'],
            "temperature": self.config['temperature'],
            "messages": [{"role": "user", "content": prompt}]
        }

        for attempt in range(self.config['max_retries']):
            try:
                async with self.session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result["choices"][0]["message"]["content"]

                        output_data = original_data.copy()
                        output_data.update({
                            'template': template,
                            'prompt': prompt,
                            'response': content
                        })

                        self.total_generations_processed += 1
                        self._save_processed_index(idx)
                        return output_data

                    elif response.status == 429:
                        await asyncio.sleep(self.config['retry_delay'])
                    else:
                        print(f"Error for idx {idx}, status: {response.status}")
                        await asyncio.sleep(self.config['retry_delay'])

            except Exception as e:
                print(f"Exception for idx {idx}: {type(e).__name__}, {str(e)}")
                await asyncio.sleep(self.config['retry_delay'])

        print(f"Max retries reached for idx {idx}. Skipping.")
        return None

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate augmented dataset using OpenAI API")
    parser.add_argument("--input", required=True, help="Input JSONL file name (should be in 'data' directory)")
    parser.add_argument("--total", type=int, default=100000, help="Total number of generations to produce")
    parser.add_argument("--batch", type=int, default=256, help="Batch size for API calls")
    parser.add_argument("--model", default="mix", help="Model to use for generation")
    parser.add_argument("--max-tokens", type=int, default=16000, help="Maximum number of tokens in the response")
    parser.add_argument("--temperature", type=float, default=1.3, help="Temperature for generation")
    parser.add_argument("--retries", type=int, default=99999999, help="Maximum number of retries for API calls")
    parser.add_argument("--delay", type=int, default=30, help="Delay between retries in seconds")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    config = {
        'input_file': args.input,
        'total_generations': args.total,
        'batch_size': args.batch,
        'model': args.model,
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'max_retries': args.retries,
        'retry_delay': args.delay,
        'api_base': os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
    }

    api = OpenAI_API(config)
    asyncio.run(api.process_jsonl_file())
