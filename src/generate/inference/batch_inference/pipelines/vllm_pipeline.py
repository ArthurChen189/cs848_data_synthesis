from vllm import LLM, SamplingParams
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import json
from tqdm import tqdm


class VLLMPipeline:
    def __init__(
        self,
        model_path: str,
        prompt_template_path: str,
        pydantic_model: BaseModel,
        aux_data_path: Optional[str] = None,
        max_context_window: int = 1536,
        max_generate_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
        num_gpus: int = 1,
        max_num_seqs: int = 30,
        batch_size: int = 100,
        verbose: bool = False,
    ):
        """Initialize the VLLMPipeline

        Args:
            model_path (str): The path to the model
            prompt_template_path (str): The path to the prompt template
            pydantic_model (BaseModel): The output format model that will be used to generate format instructions and parse the output
            aux_data_path (str, optional): The path to the auxiliary data that will be used to generate prompts
            max_context_window (int, optional): The maximum number of tokens in the context window. Defaults to 1024.
            max_generate_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1024.
            temperature (float, optional): The temperature to use for the pipeline. Defaults to 1.0.
            top_p (float, optional): The top p to use for the pipeline. Defaults to 0.95.
            num_gpus (int, optional): The number of GPUs to use for the pipeline. Defaults to 1.
            max_num_seqs (int, optional): The maximum number of sequences the GPU can handle per batch. Defaults to 10.
            batch_size (int, optional): The batch inference size. Defaults to 10.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            my_json_parser (bool, optional): Whether to use my custom JSON parser. Defaults to False.
        """

        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        
        self.temperature = temperature
        self.top_p = top_p
        self.model_path = model_path
        self.prompt_template_path = prompt_template_path
        self.pydantic_model = pydantic_model
        self.batch_size = batch_size
        self.verbose = verbose
        self.aux_data_path = aux_data_path
        
        # Initialize model
        self.sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_generate_tokens,
            stop=[]
        )

        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=max_context_window,
            tensor_parallel_size=num_gpus,
            max_num_seqs=max_num_seqs
        )

        # Initialize parser and prompt template
        self.pydantic_parser = JsonOutputParser(pydantic_object=pydantic_model)

        self.parser = self.pydantic_parser
        self.prompt_template = PromptTemplate(
            template=open(prompt_template_path).read(),
            partial_variables={"format_instructions": self.pydantic_parser.get_format_instructions()},
        )

        # Load seed data
        self.aux_data = None
        if aux_data_path:
            self.aux_data = self.preprocess_aux_data(aux_data_path)

    def preprocess_aux_data(self, aux_data_path: str) -> List[Dict[str, Any]]:
        """Override this method to implement custom preprocessing logic"""
        return json.load(open(aux_data_path))

    def generate_prompts(self, num_examples: int) -> List[str]:
        """Override this method to implement custom prompt generation logic"""
        raise NotImplementedError("Subclasses must implement generate_prompts")

    def postprocess_each_output(self, parsed_output: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Override this method to implement custom postprocessing logic"""
        return parsed_output

    def check_output_format(self, parsed_output: Dict[str, Any]) -> bool:
        """Override this method to implement custom format checking"""
        return True
    
    def get_batch_prompts_and_data(self, prompts: List[str], processed_indices: set) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Get batch prompts and data, ensuring the batch size is met if possible"""
        batch_prompts = []
        batch_data = []
        dummy_batch_data = []
        counter = 0

        for idx, prompt in enumerate(prompts):
            if idx not in processed_indices:
                batch_prompts.append(prompt)
                if self.aux_data is not None:
                    batch_data.append(self.aux_data[idx])
                else:
                    dummy_batch_data.append({"idx": idx})
                counter += 1
            
            if counter == self.batch_size:
                break
            
        if self.aux_data:
            return batch_prompts, batch_data
        else:
            # return a list consist of idx
            return batch_prompts, dummy_batch_data

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

    def run_batch_inference(
        self, 
        prompts: List[str], 
    ) -> Dict[int, Dict[str, Any]]:
        examples = {} # idx -> example
        processed_indices = set()  # Track indices of successfully processed prompts
        if self.aux_data: # check if aux data is provided
            if len(prompts) != len(self.aux_data):
                raise ValueError("Total examples must be equal to the number of prompts")
        
        with tqdm(total=len(prompts)) as pbar:
            while len(examples) < len(prompts):
                try:
                    batch_prompts, batch_data = self.get_batch_prompts_and_data(prompts, processed_indices)
                    
                    if not batch_prompts:  # If no prompts are left to process, break the loop
                        break
                    
                    outputs = self.llm.generate(batch_prompts, self.sampling_params)
                    text_outputs = [out.outputs[0].text for out in outputs]
                    
                    if len(text_outputs) != len(batch_data):
                        raise ValueError("Total examples must be equal to the number of prompts")
                    
                    for text_output, input_data in zip(text_outputs, batch_data):
                        try:
                            text_output = self.clean_text_output(text_output)
                            parsed = self.parser.parse(text_output)
                            if self.check_output_format(parsed):
                                processed = self.postprocess_each_output(parsed, input_data)
                                examples[input_data["idx"]] = processed
                                processed_indices.add(input_data["idx"])  # Mark this prompt as processed
                                pbar.update(1)
                        except Exception as e:
                            if self.verbose:
                                print(f"Parsing error: {e}")
                            continue                    
                except Exception as e:
                    if self.verbose:
                        print(f"Batch inference error: {e}")
                    continue
        return examples

    def generate(
        self, 
        num_examples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Main pipeline execution"""
        
        # 1. Generate prompts
        print(f"Generating {num_examples} prompts")
        prompts = self.generate_prompts(num_examples)

        # print one prompt for validation
        print(f"Below is a sample prompt\n--------------------------------")
        print(prompts[0])
        print(f"--------------------------------")  

        # 2. Run batch inference and postprocess results
        print(f"Running batch inference for {len(prompts)} prompts")
        results = self.run_batch_inference(prompts)
        
        # 3. postprocess all results
        results = self.postprocess_all_results(results)

        return results

    def postprocess_all_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Override this method to implement custom postprocessing logic"""
        return results

    def save_results(
        self,
        examples: List[Dict[str, Any]],
        output_path: str,
        metadata_path: Optional[str] = None,
        metadata: Dict[str, Any] = {}
    ):
        with open(output_path, "w") as f:
            json.dump(examples, f, indent=2)
        if metadata_path:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)