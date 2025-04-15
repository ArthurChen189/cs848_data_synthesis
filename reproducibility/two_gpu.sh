NUM_RUNS=10

for i in $(seq 1 $NUM_RUNS); do
    python -m src.generate.sentiment_analysis_synthesis \
        --num_examples 1000 \
        --model_path Qwen/Qwen2.5-1.5B-Instruct \
        --output_folder data/benchmarking-batch \
        --use_clean_text_output \
        --prompt_template_path ./prompt_templates/sentiment_analysis/zero-shot_bg_v1.txt \
        --num_gpus 2 \
        --max_num_seqs 1000 \
        --batch_size 1000 \
        --benchmark_output_folder data/benchmarking-batch/two_gpu_benchmarking
done