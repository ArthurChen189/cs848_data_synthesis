from .vllm_pipeline import VLLMPipeline
from ..data_models import SentimentAnalysisModel
from typing import List, Dict, Any, Optional
import json
import re
import numpy as np
import os
from pathlib import Path
import sys

ROOT_DATA_PATH ="./data/sentiment_analysis"
DS_TRAIN_PATH = os.path.join(ROOT_DATA_PATH, "cleaned_ds_train.json")
DS_VALIDATION_PATH = os.path.join(ROOT_DATA_PATH, "cleaned_ds_validation.json")

class SentimentAnalysisSynthesisPipeline(VLLMPipeline):
    def __init__(
        self,
        model_path: str,
        prompt_template_path: str,
        num_shots: int = 3,
        num_tickers: int = 1,
        original_data_path: str = None,
        max_context_window: int = 1024,
        max_generate_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.95,
        **kwargs
    ):
        super().__init__(model_path, prompt_template_path, SentimentAnalysisModel, original_data_path, 
                         max_context_window, max_generate_tokens, temperature, top_p, **kwargs)
        self.num_shots = num_shots
        self.num_tickers = num_tickers
        self.train_stock_tickers, self.test_stock_tickers = self._get_stock_tickers()

    def _get_stock_tickers(self) -> tuple[list[str], list[str]]:
        """Extract stock tickers from training and validation data"""
        with open(DS_TRAIN_PATH, "r") as f:
            train_data = json.load(f)
        with open(DS_VALIDATION_PATH, "r") as f:
            validation_data = json.load(f)

        def extract_tickers(data):
            tickers = []
            for item in data:
                x = item['text']
                x = re.sub(r'https?://\S+', '', x)
                stock_ticket = re.search(r'\$([A-Z]+)', x)
                if stock_ticket:
                    tickers.append(stock_ticket.group(1))
            return tickers

        return extract_tickers(train_data), extract_tickers(validation_data)

    def _format_stock_tickers(self, stock_tickers: Optional[List[str]]) -> str:
        """Format stock tickers for prompt generation"""
        if not stock_tickers:
            return ""
        sampled_tickers = np.random.choice(stock_tickers, self.num_tickers, replace=False)
        return ', '.join([f'${ticker}' for ticker in sampled_tickers])

    def _sample_few_shot_examples(self, train_data: List[Dict[str, Any]]) -> str:
        """Sample few-shot examples from training data ensuring diverse labels"""
        examples = []
        y_labels = set()
        while len(examples) < self.num_shots:
            indices = np.random.choice(len(train_data), self.num_shots, replace=False)
            for idx in indices:
                x = train_data[int(idx)]['text']
                y = train_data[int(idx)]['label']
                if y not in y_labels:
                    examples.append({"headline": x, "sentiment": y})
                    y_labels.add(y)
        return json.dumps(examples, indent=2)

    def generate_prompts(self, num_examples: int) -> List[str]:
        """Generate prompts based on the prompt type (zero-shot or few-shot)"""
        prompts = []
        
        # Determine which stock tickers to use based on prompt type
        stock_tickers = None
        if 'train-time-info' in self.prompt_template_path:
            stock_tickers = self.train_stock_tickers
        elif 'test-time-info' in self.prompt_template_path:
            stock_tickers = self.test_stock_tickers

        if "zero-shot" in self.prompt_template_path:
            inputs_list = [{"stock_ticker": self._format_stock_tickers(stock_tickers)} for _ in range(num_examples)]
            prompts = [self.prompt_template.invoke(inputs).text for inputs in inputs_list]
        
        elif "few-shot" in self.prompt_template_path:
            # Load training data for few-shot examples
            with open(DS_TRAIN_PATH, "r") as f:
                train_data = json.load(f)
            
            inputs_list = [
                {"examples": self._sample_few_shot_examples(train_data), 
                 "stock_ticker": self._format_stock_tickers(stock_tickers)
                } for _ in range(num_examples)
            ]
            prompts = [self.prompt_template.invoke(inputs).text for inputs in inputs_list]
        
        else:
            raise ValueError(f"Invalid prompt template path: {self.prompt_template_path}")
        
        prompts = [sys.intern(p) for p in prompts]
        return prompts

    def check_output_format(self, parsed_output: Dict[str, Any]) -> bool:
        """Verify the output format matches expected structure"""
        if isinstance(parsed_output, list):
            for item in parsed_output:
                if not isinstance(item, dict):
                    return False
                if set(item.keys()) != set(['headline', 'sentiment']):
                    return False                
                if re.search(r'[a-zA-Z]+', item['headline']) is None:
                    return False
            return True
        elif isinstance(parsed_output, dict):
            if set(parsed_output.keys()) != set(['headline', 'sentiment']):
                return False
            if re.search(r'[a-zA-Z]+', parsed_output['headline']) is None:
                return False
            return True
        else:
            return False
    
    def postprocess_all_results(self, results: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Postprocess all results"""
        final_results = []
        for result in results.values():
            final_results.extend(result)
        return final_results

    def save_results(self, examples: List[Dict[str, Any]], output_path: Path, metadata_path: Path, metadata: Dict[str, Any]):
        """Save the generated examples and metadata"""
        # Save examples
        with open(output_path, 'w') as f:
            json.dump(examples, f, indent=2)
        
        metadata = {**metadata, "temperature": self.temperature, "top_p": self.top_p}
        # create a new metadata folder
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
