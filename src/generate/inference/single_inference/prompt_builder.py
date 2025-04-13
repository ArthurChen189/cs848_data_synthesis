class PromptBuilder:
    def __init__(self, model, url='http://localhost:11434', 
                 service='ollama', api_key="", temperature=1.0, top_p=0.95, max_tokens=512):
        if service == 'ollama':
            from langchain_ollama import OllamaLLM
            self.model = OllamaLLM(model=model, base_url=url, keep_alive=60, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        elif service == 'cerebras':
            from langchain_cerebras import ChatCerebras
            self.model = ChatCerebras(model=model, api_key=api_key, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        elif service == 'vllm':
            from langchain_community.llms import VLLM
            self.model = VLLM(model=model, 
                trust_remote_code=True,
                vllm_kwargs={"max_model_len": max_tokens, "gpu_memory_utilization": 0.95, "max_model_len": max_tokens},
                temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        elif service == 'nebius':
            from langchain_openai import ChatOpenAI
            self.model = ChatOpenAI(model=model, api_key=api_key, temperature=temperature, top_p=top_p, max_tokens=max_tokens, 
                                    base_url=url, model_kwargs={ "response_format": { "type": "json_object" } })
        else:
            raise NotImplementedError(f'No support for service {service}')