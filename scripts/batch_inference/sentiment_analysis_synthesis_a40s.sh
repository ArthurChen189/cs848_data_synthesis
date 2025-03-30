#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=sentiment_analysis_synthesis
#SBATCH --gres=gpu:a40:4
#SBATCH --time=2:00:00
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH --qos=m4
#SBATCH --partition=a40
#SBATCH --output=logs/sentiment_analysis_synthesis_4gpu_%j.out
#SBATCH --error=logs/sentiment_analysis_synthesis_4gpu_%j.err

source /fs01/home/arthur/.zshrc
eval "$(micromamba shell hook --shell zsh)"
micromamba activate data-synthesis
cd /h/arthur/Workspace/data_synthesis

SCRIPT=src/generate/sentiment_analysis_synthesis.py
ROOT_PROMPT_TEMPLATE_PATH=./prompt_templates/sentiment_analysis
ZERO_SHOT_PROMPTS=("zero-shot_v1.txt" "zero-shot_bg_v1.txt" "zero-shot_bg_train-time-info_v1.txt" "zero-shot_bg_test-time-info_v1.txt")
FEW_SHOT_PROMPTS=("few-shot_v1.txt" "few-shot_bg_v1.txt" "few-shot_bg_train-time-info_v1.txt" "few-shot_bg_test-time-info_v1.txt")

# inference args
NUM_GPU=4
MAX_NUM_SEQS=30
BATCH_INFERENCE_SIZE=100
NUM_EXAMPLES=500 # we want 1000 examples but each prompt is asked to generate 3 examples, so 500 prompts are enough

# input args
model=$1
prompt=$2

if [ "$prompt" == "zero-shot" ]; then
    PROMPTS=("${ZERO_SHOT_PROMPTS[@]}")
    echo "Executing zero-shot prompts"
elif [ "$prompt" == "few-shot" ]; then
    PROMPTS=("${FEW_SHOT_PROMPTS[@]}")
    echo "Executing few-shot prompts"
elif [ "$prompt" == "all" ]; then
    PROMPTS=("${ZERO_SHOT_PROMPTS[@]}" "${FEW_SHOT_PROMPTS[@]}")
    echo "Executing all prompts"
else
    echo "Invalid prompt: $prompt"
    exit 1
fi

# args
echo "MODEL: $model"
for prompt in "${PROMPTS[@]}"; do
    echo "PROMPT: $prompt"
    PYTHONPATH=/h/arthur/Workspace/data_synthesis python $SCRIPT \
        --num_examples $NUM_EXAMPLES \
        --model_path $model \
        --prompt_template_path $ROOT_PROMPT_TEMPLATE_PATH/$prompt \
        --num_gpus $NUM_GPU \
        --batch_size $BATCH_INFERENCE_SIZE \
        --max_num_seqs $MAX_NUM_SEQS
done