python -m src.generate.sentiment_analysis_synthesis \
    --num_examples 1000 \
    --model_path Qwen/Qwen2.5-1.5B-Instruct \
    --output_folder data/benchmarking-batch \
    --use_clean_text_output \
    --prompt_template_path ./prompt_templates/sentiment_analysis/zero-shot_bg_v1.txt \
    --verbose \
    --num_gpus 1 \
    --benchmark 