# CS848 Data Synthesis Project

This repository contains code for generating synthetic sentiment analysis data using large language models. The project is part of CS848 coursework and focuses on creating high-quality synthetic datasets for sentiment analysis tasks.

## Project Structure

```
.
├── data/               # Data directory for storing generated datasets
├── prompt_templates/   # Templates for LLM prompts
├── scripts/           # Utility scripts
├── src/               # Source code
│   └── generate/      # Data generation code
│       ├── inference/ # Inference pipeline
│       └── utils.py   # Utility functions
└── requirements.txt   # Project dependencies
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Using vLLM Backend
The main script for generating synthetic data is `src/generate/sentiment_analysis_synthesis.py`. Here's how to use it:

```bash
python src/generate/sentiment_analysis_synthesis.py \
    --model_path <path_to_model> \
    --prompt_template_path <path_to_template> \
    --output_folder <output_directory> \
    --num_examples <number_of_examples> \
    [--num_shots <number_of_shots>] \
    [--num_tickers <number_of_tickers>] \
    [--num_gpus <number_of_gpus>] \
    [--batch_size <batch_size>] \
    [--max_context_window <window_size>] \
    [--max_generate_tokens <max_tokens>] \
    [--max_num_seqs <max_sequences>] \
    [--verbose]
```

```
python3 -m src.generate.sentiment_analysis_synthesis \
  --model_path Qwen/Qwen2.5-7B-Instruct-AWQ \
  --prompt_template_path ./prompt_templates/sentiment_analysis/few-shot_bg_test-time-info_v1.txt \
  --num_examples 1 \
  --output_folder ./test_output
```
#### Key Parameters

- `model_path`: Path to the language model to use
- `prompt_template_path`: Path to the prompt template file
- `output_folder`: Directory to save generated data
- `num_examples`: Number of examples to generate
- `num_shots`: Number of few-shot examples (optional)
- `num_tickers`: Number of tickers to use (optional)
- `num_gpus`: Number of GPUs to use (optional)
- `batch_size`: Batch size for generation (optional)
- `max_context_window`: Maximum context window size (optional)
- `max_generate_tokens`: Maximum tokens to generate (optional)
- `max_num_seqs`: Maximum number of sequences (optional)
- `verbose`: Enable verbose output (optional)

### Using Ollama, API, or other backends
The main script for generating synthetic data is `src/generate/inference/single_inference/llmgen.py`. Here's how to use it:

If you are serving LLMs locally via vLLM or Ollama, first you need to start the server in a separate terminal:
```bash
ollama run <ollama_model_name> 
```
or
```bash
vllm --port <port_number>
```

Then you can run the following command to generate synthetic data:
```bash
python src/generate/inference/single_inference/llmgen.py \
    --prompt <path_to_prompt_template> \
    --model <model_name> \
    --service <service_name> \
    --num_examples <number_of_examples> \
    [--output_path <output_directory>] \
    [--resume_path <path_to_resume_file>] \
    [--num_tickers <number_of_tickers>] \
    [--base_url <base_url>] \
    [--deduplicate] \
    [--api_limit] \
    [--verbose]
```

Example:
```bash
python src/generate/inference/single_inference/llmgen.py \
    --prompt ./prompt_templates/sentiment_analysis/few-shot_bg_test-time-info_v1.txt \
    --model llama3.1-8b \
    --service ollama \
    --num_examples 100 \
    --output_path ./data/sentiment_analysis/synthetic_data \
    --num_tickers 2 \
    --base_url http://localhost:11434 \ # ollama default base url
    --deduplicate \
    --verbose
```

#### Key Parameters

- `--prompt`: Path to the prompt template file (required)
- `--model`: Name of the model to use (required)
- `--service`: Service to use for generation (choices: "cerebras", "ollama", "vllm", "nebius", default: "cerebras")
- `--num_examples`: Number of examples to generate (default: 1000)
- `--output_path`: Directory to save generated data (default: "./data/sentiment_analysis")
- `--resume_path`: Path to a JSON file to resume generation from (optional)
- `--num_tickers`: Number of stock tickers to use to guide generation (default: 1)
- `--deduplicate`: Flag to deduplicate generated examples (default: False)
- `--api_limit`: Flag to limit API requests per minute (default: False)
- `--verbose`: Flag to enable verbose output (default: False)

#### Supported Services

- **Ollama**: Local LLM service
- **Cerebras**: Cloud-based LLM service
- **Nebius**: Cloud-based LLM service
- **vLLM**: Local LLM service with GPU acceleration

#### Configuration

The script uses a `secrets.json` file for API keys and endpoints. Make sure this file exists in the project root with the following structure:

```json
{
  "cerebras": {
    "api_key_arthur": "your_api_key"
  },
  "nebius": {
    "api_key": "your_api_key",
    "api_endpoint": "your_api_endpoint"
  },
  "ollama": {
    "api_endpoint": "http://localhost:11434"
  }
}
```

## Dependencies

The project relies on several key libraries:
- datasets
- sentence-transformers
- langchain and related packages
- vllm
- tqdm
- func-timeout

See `requirements.txt` for the complete list of dependencies.

## Output

The generated data will be saved in the specified output directory with the following naming convention:
```
<model_name>_<num_examples>_prompt=<template_name>.json
```

Metadata about the generation process is also saved in a separate metadata directory.

## License

This project is part of CS848 coursework. Please refer to the course guidelines for usage and distribution terms.
