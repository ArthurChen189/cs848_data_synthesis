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

### Key Parameters

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
