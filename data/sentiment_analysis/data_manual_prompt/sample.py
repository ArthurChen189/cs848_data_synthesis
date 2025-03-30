from pathlib import Path
import json
import numpy as np
import re

SEED = 42
SAMPLE_SIZE = 1000
FULL_DATA_PATH = Path('./data/sentiment_analysis/data_manual_prompt/synthetic_data_full')
OUTPUT_PATH = Path(f'./data/sentiment_analysis/data_manual_prompt/synthetic_data_sampled={SAMPLE_SIZE}')

np.random.seed(SEED)

def check_format(item):
    if re.search(r'[a-zA-Z]+', item['headline']) is None:
        return False
    return True


for path in FULL_DATA_PATH.glob('*prompt=*.json'):
    with open(path, 'rt') as f:
        data = [item for item in json.load(f) if check_format(item)]
    print(f"Loaded {len(data)} examples from {path}")
    
    # sample
    data = np.random.choice(data, SAMPLE_SIZE, replace=False)
    print(f"Sampled {len(data)} examples")
    
    # save
    output_path = OUTPUT_PATH / path.name
    if output_path.exists():
        print(f"Skipping {output_path} because it already exists")
        continue
    with open(output_path, 'wt') as f:
        json.dump(list(data), f, indent=2)
    
    
    