mm activate data-synthesis
cd /home/arthur/Workplace/R2L/data_synthesis

SHOTS=3
TICKERS=1

## Few-shot generation
# qwen2.5:7b
python llmgen.py --prompt ./prompt_templates/sentiment_analysis/few-shot_v1.txt --service ollama --model qwen2.5:7b
python llmgen.py --prompt ./prompt_templates/sentiment_analysis/few-shot_bg_v1.txt --service ollama --model qwen2.5:7b
python llmgen.py --prompt ./prompt_templates/sentiment_analysis/few-shot_bg_train-time-info_v1.txt --service ollama --model qwen2.5:7b
python llmgen.py --prompt ./prompt_templates/sentiment_analysis/few-shot_bg_test-time-info_v1.txt --service ollama --model qwen2.5:7b

# qwen2.5:32b
python llmgen.py --prompt ./prompt_templates/sentiment_analysis/few-shot_v1.txt --service ollama --model qwen2.5:32b
python llmgen.py --prompt ./prompt_templates/sentiment_analysis/few-shot_bg_v1.txt --service ollama --model qwen2.5:32b
python llmgen.py --prompt ./prompt_templates/sentiment_analysis/few-shot_bg_train-time-info_v1.txt --service ollama --model qwen2.5:32b
python llmgen.py --prompt ./prompt_templates/sentiment_analysis/few-shot_bg_test-time-info_v1.txt --service ollama --model qwen2.5:32b

# llama3.1-8b
python llmgen.py --prompt ./prompt_templates/sentiment_analysis/few-shot_v1.txt --service cerebras --model llama3.1-8b
python llmgen.py --prompt ./prompt_templates/sentiment_analysis/few-shot_bg_v1.txt --service cerebras --model llama3.1-8b
python llmgen.py --prompt ./prompt_templates/sentiment_analysis/few-shot_bg_train-time-info_v1.txt --service cerebras --model llama3.1-8b
python llmgen.py --prompt ./prompt_templates/sentiment_analysis/few-shot_bg_test-time-info_v1.txt --service cerebras --model llama3.1-8b

# llama3.3-70b
python llmgen.py --prompt ./prompt_templates/sentiment_analysis/few-shot_v1.txt --service cerebras --model llama3.3-70b
python llmgen.py --prompt ./prompt_templates/sentiment_analysis/few-shot_bg_v1.txt --service cerebras --model llama3.3-70b
python llmgen.py --prompt ./prompt_templates/sentiment_analysis/few-shot_bg_train-time-info_v1.txt --service cerebras --model llama3.3-70b
python llmgen.py --prompt ./prompt_templates/sentiment_analysis/few-shot_bg_test-time-info_v1.txt --service cerebras --model llama3.3-70b


