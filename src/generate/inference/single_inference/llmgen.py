import sys
import os

# Adds the parent of src/ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from src.generate.inference.batch_inference.data_models import SentimentAnalysisModel
from src.generate.inference.single_inference.prompt_builder import PromptBuilder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import json
import re
import time
import numpy as np
from tqdm import tqdm
import argparse
from typing import Dict, Any
from pathlib import Path
from src.generate.utils import get_model_name
from time import time
from datetime import datetime

# Metadata
MAX_TOKENS = 1024
TEMPERATURE = 1.0
TOP_P = 0.95
SHOTS = 3

def get_stock_tickers():
    with open("./data/sentiment_analysis/cleaned_ds_train.json", "r") as f:
        train_data = json.load(f)

    with open("./data/sentiment_analysis/cleaned_ds_validation.json", "r") as f:
        validation_data = json.load(f)

    train_stock_tickets = []
    test_stock_tickets = []

    for _, item in enumerate(train_data):
        x = item['text']
        
        # remove urls
        x = re.sub(r'https?://\S+', '', x)

        # extract the stock ticket (i.e., after $ symbol)
        stock_ticket = re.search(r'\$([A-Z]+)', x)
        if stock_ticket:
            stock_ticket = stock_ticket.group(1)
            train_stock_tickets.append(stock_ticket)

    for _, item in enumerate(validation_data):
        x = item['text']
        
        # remove urls
        x = re.sub(r'https?://\S+', '', x)

        # extract the stock ticket (i.e., after $ symbol)
        stock_ticket = re.search(r'\$([A-Z]+)', x)
        if stock_ticket:
            stock_ticket = stock_ticket.group(1)
            test_stock_tickets.append(stock_ticket)

    return train_stock_tickets, test_stock_tickets


def format_stock_tickers(stock_tickets, num_tickers=1):
    if stock_tickets is None:
        return ""
    sampled_stock_tickers = np.random.choice(stock_tickets, num_tickers)
    output_string = ', '.join([f'${ticker}' for ticker in sampled_stock_tickers])
    return output_string

def check_output_format(parsed_output: Dict[str, Any]) -> bool:
    """Verify the output format matches expected structure"""
    if isinstance(parsed_output, list):
        for item in parsed_output:
            if not isinstance(item, dict):
                return False
            if set(item.keys()) != set(['headline', 'sentiment']):
                return False
            if str(item['sentiment']) not in ['0', '1', '2']:
                return False
            if re.search(r'[a-zA-Z]+', item['headline']) is None:
                return False
        return True
    return False

def clean_text_output(self, text_output: str) -> str:
        """Clean the text output"""
        # First extract content between triple backticks if present
        if "```" in text_output:
            parts = text_output.split("```")
            # Get the content between first and second ```
            if len(parts) >= 3:
                text_output = parts[1].strip()
        else:
            text_output = text_output.strip()
            if text_output.lower().startswith('json'):
                text_output = text_output[len("json"):].strip()
            return text_output
        
        # Remove language identifier if it starts with 'j'
        if text_output.lower().startswith('json'):
            text_output = text_output[len("json"):].strip()
        
        return text_output


def generate_zero_shot(chain, num_examples, stock_tickers, args, resume_data=None):
    zero_shot_examples = resume_data if resume_data else []
    print(f"Generating {num_examples} zero-shot examples, using {args.model} model, prompt: {args.prompt}")
    start_bar = len(zero_shot_examples)
    with tqdm(total=num_examples, initial=start_bar) as pbar:
        while len(zero_shot_examples) < num_examples:
            try:
                output_text = chain.invoke({"stock_ticker": format_stock_tickers(stock_tickers, args.num_tickers)})
                
                try:
                    parser = JsonOutputParser(pydantic_object=SentimentAnalysisModel)
                    # output_text = clean_text_output(output_text)
                    out = parser.parse(output_text)
                except Exception as parse_err:
                    # Failure counter logic
                    print("failed")
                    count_file = "fail"
                    if not os.path.exists(count_file):
                        with open(count_file, "w") as f:
                            f.write("1")
                    else:
                        with open(count_file, "r+") as f:
                            count = int(f.read().strip())
                            f.seek(0)
                            f.write(str(count + 1))
                            f.truncate()
                    continue  # skip to next iteration if parsing fails

                if check_output_format(out):
                    pbar.update(len(out))
                    zero_shot_examples.extend(out)
                    # Success counter logic
                    count_file = "suc"
                    if not os.path.exists(count_file):
                        with open(count_file, "w") as f:
                            f.write("1")
                    else:
                        with open(count_file, "r+") as f:
                            count = int(f.read().strip())
                            f.seek(0)
                            f.write(str(count + 1))
                            f.truncate()
                else:
                    count_file = "fail"
                    if not os.path.exists(count_file):
                        with open(count_file, "w") as f:
                            f.write("1")
                    else:
                        with open(count_file, "r+") as f:
                            count = int(f.read().strip())
                            f.seek(0)
                            f.write(str(count + 1))
                            f.truncate()

                if args.api_limit:
                    time.sleep(2)

            except Exception as e:
                if 'exceeded' in str(e) and 'limit' in str(e):
                    if 'minute' in str(e):
                        print(f"error: {e}")
                        time.sleep(60)
                    else:
                        print(f"error: {e}")
                        break
                if args.verbose:
                    print(f"error: {e}")
                continue

    return zero_shot_examples

def generate_few_shot(chain, num_examples, cleaned_train_real, stock_tickers, args, resume_data=None):
    def sample_examples(cleaned_train_real, n_examples):
        examples = []
        y_labels = set()
        while len(examples) < n_examples:
            indices = np.random.choice(len(cleaned_train_real), n_examples, replace=False)
            for idx in indices:
                x = cleaned_train_real[int(idx)]['text']
                y = cleaned_train_real[int(idx)]['label']
                if y not in y_labels:
                    examples.append({"x": x, "y": y})
                    y_labels.add(y)

        # convert to json string
        examples_json = json.dumps(examples, indent=2)
        return examples_json
    
    few_shot_examples = resume_data if resume_data else []
    with tqdm(total=num_examples, initial=len(few_shot_examples)) as pbar:
        while len(few_shot_examples) < num_examples:
            try:
                output_text = chain.invoke({
                    "examples": sample_examples(cleaned_train_real, SHOTS), 
                    "stock_ticker": format_stock_tickers(stock_tickers, args.num_tickers)
                })

                try:
                    out = parser.parse(output_text)
                    # Success counter logic
                    count_file = "suc"
                    if not os.path.exists(count_file):
                        with open(count_file, "w") as f:
                            f.write("1")
                    else:
                        with open(count_file, "r+") as f:
                            count = int(f.read().strip())
                            f.seek(0)
                            f.write(str(count + 1))
                            f.truncate()
                except Exception as parse_err:
                    if args.verbose:
                        print(f"Parse error: {parse_err}")
                    # Failure counter logic
                        count_file = "fail"
                        if not os.path.exists(count_file):
                            with open(count_file, "w") as f:
                                f.write("1")
                        else:
                            with open(count_file, "r+") as f:
                                count = int(f.read().strip())
                                f.seek(0)
                                f.write(str(count + 1))
                                f.truncate()
                    continue  # skip to next iteration if parsing fails

                if check_output_format(out):
                    pbar.update(len(out))
                    few_shot_examples.extend(out)

                if args.api_limit:
                    time.sleep(2)  # since cerebras has a limit of 30 requests per minute

            except Exception as e:
                if 'exceeded' in str(e) and 'limit' in str(e):
                    if 'minute' in str(e):
                        print(f"error: {e}")
                        time.sleep(60)
                    else:
                        print(f"error: {e}")
                        break
                if args.verbose:
                    print(f"error: {e}")
                continue

    return few_shot_examples


def get_metadata(args):
    # store all the metadata
    metadata = {
        "MODEL": args.model,
        "MAX_TOKENS": MAX_TOKENS,
        "TEMPERATURE": TEMPERATURE,
        "TOP_P": TOP_P,
        "NUM_TICKERS": args.num_tickers,
        "NUM_SHOTS": SHOTS,
        "PROMPT": args.prompt
    }
    return metadata


def deduplicate(examples):
    new_examples = []
    seen_x = set()
    for item in examples:
        if item['headline'] in seen_x:
            continue
        seen_x.add(item['headline'])
        new_examples.append(item)
    return new_examples


def save(examples, args):
    """Save generated examples and metadata to JSON files.
    
    Args:
        examples: List of generated examples
        args: Command line arguments
    """
    # Deduplicate examples if requested
    if args.deduplicate:
        examples = deduplicate(examples)
    
    # Normalize model name for file naming
    args.model = get_model_name(args.model)
    
    # Extract prompt name from path for tagging
    prompt_name = Path(args.prompt).stem
    tags = f"prompt={prompt_name}"
    
    # Define output paths
    output_folder = Path(args.output_path)
    metadata_folder = output_folder / 'metadata'

    if not metadata_folder.exists():
        metadata_folder.mkdir(parents=True, exist_ok=True)
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    fpath = output_folder / f'{args.model}_{len(examples)}_{tags}.json'
    metapath = metadata_folder / f'{args.model}_{tags}_metadata.json'
    
    # Log save information
    print(f"Model: {args.model}, num_examples: {len(examples)}, saving to {fpath}")
    
    # Save examples and metadata
    for path, data in [(fpath, examples), (metapath, get_metadata(args))]:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


def main(args):
    parser = JsonOutputParser(pydantic_object=SentimentAnalysisModel)

    if args.service == "cerebras":
        # API keys
        API_INFO = json.load(open("./secrets.json"))
        builder = PromptBuilder(service='cerebras', api_key=API_INFO["cerebras"]["api_key_arthur"], 
                                model=args.model, temperature=TEMPERATURE, 
                                top_p=TOP_P, max_tokens=MAX_TOKENS)
    elif args.service == "nebius":
        # API keys
        API_INFO = json.load(open("./secrets.json"))
        builder = PromptBuilder(service='nebius', api_key=API_INFO["nebius"]["api_key"], 
                                url=API_INFO["nebius"]["api_endpoint"],
                                model=args.model, temperature=TEMPERATURE, 
                                top_p=TOP_P, max_tokens=MAX_TOKENS)
    elif args.service == "ollama":
        builder = PromptBuilder(service='ollama', url=args.base_url,
                                model=args.model, temperature=TEMPERATURE, 
                                top_p=TOP_P, max_tokens=MAX_TOKENS)
    elif args.service == "vllm":
        builder = PromptBuilder(service='vllm', url=args.base_url,
                                model=args.model, temperature=TEMPERATURE, 
                                top_p=TOP_P, max_tokens=MAX_TOKENS)
    else:
        raise ValueError(f"Invalid service: {args.service}")
    
    prompt = PromptTemplate(
        template=open(args.prompt).read(),
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | builder.model

    # get stock tickers
    train_stock_tickets, test_stock_tickets = get_stock_tickers()
    stock_tickers = None 
    if 'train-time-info' in args.prompt:
        stock_tickers = train_stock_tickets
    elif 'test-time-info' in args.prompt:
        stock_tickers = test_stock_tickets   
    
    # load resume data if provided
    resume_data = None
    if args.resume_path:
        dataset_name = Path(args.resume_path).stem.split("prompt=")[-1]
        prompt_name = Path(args.prompt).stem
        assert dataset_name == prompt_name, f"dataset name and prompt name must match, {dataset_name} != {prompt_name}"
        with open(args.resume_path, "r") as f:
            resume_data = json.load(f)
        print(f"Resuming from {args.resume_path}, loaded {len(resume_data)} examples")

    # generate examples
    print(f"Generating {args.num_examples} examples, using {args.model} model, prompt: {args.prompt}")
    start_time = time()
    if "zero-shot" in args.prompt:
        examples = generate_zero_shot(chain, args.num_examples, stock_tickers, args, resume_data)
        save(examples, args)
    elif "few-shot" in args.prompt:
        # load cleaned_train_real
        with open("./data/sentiment_analysis/cleaned_ds_train.json", "r") as f:
            cleaned_train_real = json.load(f)

        examples = generate_few_shot(chain, args.num_examples, cleaned_train_real, stock_tickers, args, resume_data)
        save(examples, args)
    else:
        raise ValueError(f"Invalid prompt: {args.prompt}")

    end_time = time()
    # Calculate throughput
    total_time = end_time - start_time
    throughput = args.num_examples / total_time

    # Prepare benchmarking results
    benchmarking_results = {
        "total_time": total_time,
        "throughput": throughput,
        "num_examples": args.num_examples,
        "model": args.model,
        "prompt": args.prompt,
        "service": args.service
    }

    # Save benchmarking results to a json file
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_output_path = Path(args.output_path) / f"benchmarking_results_{date}.json"
    with open(benchmark_output_path, "w") as f:
        json.dump(benchmarking_results, f, indent=2)

    print(f"Benchmarking results saved to {benchmark_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data parameters
    parser.add_argument("--prompt", type=str, required=True, help="path to the prompt file")
    parser.add_argument("--output_path", type=str, default="./data/sentiment_analysis/", help="path to save the generated examples")
    parser.add_argument("--resume_path", type=str, default=None, help="path of the json file to resume from")

    # generation parameters
    parser.add_argument("--num_examples", type=int, default=1000, help="number of examples to generate")
    parser.add_argument("--num_tickers", type=int, default=1, help="number of stock tickers (e.g., $AAPL, $TSLA, etc.) to use to guide the generation")
    parser.add_argument("--deduplicate", type=bool, default=False, help="whether to deduplicate the generated examples")

    # model parameters
    parser.add_argument("--service", type=str, default="cerebras", choices=["cerebras", "ollama", "vllm", "nebius"], help="host service to use for generation")
    parser.add_argument("--model", type=str, default="llama3.1-8b", required=True, help="model to use for generation")
    parser.add_argument("--api_limit", action="store_true", help="whether to constraint the number of requests per minute")
    parser.add_argument("--verbose", action="store_true", help="whether to print verbose output")
    parser.add_argument("--base_url", type=str, default="http://localhost:11434", help="base url to use for ollama")
    args = parser.parse_args()
    main(args)