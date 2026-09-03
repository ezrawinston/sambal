#!/bin/bash

# Configuration
CONFIG_FILE="../configs/small.json"
TOKENIZER_PATH="../gpt-bert-babylm-small/tokenizer.json"
MAX_NEW_TOKENS=100
TEMPERATURE=0.8
REPETITION_PENALTY=1

# Models
MODELS=(
    "../trained_models/gptbert_sambal_long_ema.bin"
    "../trained_models/gptbert_babycosmofine_long_ema.bin"
)
MODEL_NAMES=("sambal" "standard")

# LoRA adapters (corresponding to each model)
LORA_ADAPTERS=(
    "ft_out/lotr_lora_r_32_a_16_sambal/best_ppl_trainable_params.pt"
    "ft_out/lotr_lora_r_32_a_16_gptbert/best_ppl_trainable_params.pt"
)
LORA_R=32
LORA_ALPHA=16
LORA_SCOPE="attn,mlp"

# Prompts
PROMPTS=(
    "\`The dark fire will not avail you, flame of Udyn. Go back to the Shadow! You cannot pass.' The Balrog made no answer."
    "\`It's a dangerous business, Frodo, going out of your door' he used to say."
)
PROMPT_NAMES=("balrog" "biz")

# Seeds
SEEDS=(0 42)

# Loop over all combinations
for model_idx in "${!MODELS[@]}"; do
    MODEL="${MODELS[$model_idx]}"
    MODEL_NAME="${MODEL_NAMES[$model_idx]}"
    LORA_ADAPTER="${LORA_ADAPTERS[$model_idx]}"

    for prompt_idx in "${!PROMPTS[@]}"; do
        PROMPT="${PROMPTS[$prompt_idx]}"
        PROMPT_NAME="${PROMPT_NAMES[$prompt_idx]}"

          echo "=============================================="
          echo "Model: $MODEL_NAME | Prompt: $PROMPT_NAME | Seed: $SEED | LoRA: NO"
          echo "=============================================="
          python gen_text_beam.py \
              --config_file "$CONFIG_FILE" \
              --tokenizer_path "$TOKENIZER_PATH" \
              --checkpoint "$MODEL" \
              --prompt "$PROMPT" \
              --strategy beam \
              --num_beams 8 \
              --length_penalty 0.6 \
              --early_stopping \
              --repetition_penalty 1.5 \
              --max_new_tokens 100

          echo ""
          echo "=============================================="
          echo "Model: $MODEL_NAME | Prompt: $PROMPT_NAME | Seed: $SEED | LoRA: YES"
          echo "=============================================="
          python gen_text_beam.py \
              --config_file "$CONFIG_FILE" \
              --tokenizer_path "$TOKENIZER_PATH" \
              --checkpoint "$MODEL" \
              --lora_adapter "$LORA_ADAPTER" \
              --lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_scope "$LORA_SCOPE" \
              --train_embeddings \
              --prompt "$PROMPT" \
              --strategy beam \
              --num_beams 8 \
              --length_penalty 0.6 \
              --early_stopping \
              --repetition_penalty 1.5 \
              --max_new_tokens 100

            echo ""
    done
done