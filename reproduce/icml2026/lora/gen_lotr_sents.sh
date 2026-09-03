#!/bin/bash
# LotR continuation examples: sampled generations from both long models,
# with and without the LoRA adapter, over both paper prompts and the fixed
# seed list. Run from lm/gpt-bert/pretraining. Outputs are stochastic
# sampling draws; the paper prints a selection of them.

# Configuration
CONFIG_FILE="../configs/small.json"
TOKENIZER_PATH="../gpt-bert-babylm-small/tokenizer.json"
MAX_NEW_TOKENS=75
TEMPERATURE=0.9
REPETITION_PENALTY=1.25

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
"\`The dark fire will not avail you, flame of Udyn. Go back to the Shadow! You cannot pass,' bellowed Gandalf."
"\`It's a dangerous business, Frodo, going out of your door' he used to say."
)
PROMPT_NAMES=("balrog" "biz")

# Seeds
SEEDS=(983721 625431 322131 872631 542313 221031 987213 6534121 3213211 8732611 5432131 2132011)

# Loop over all combinations
for model_idx in "${!MODELS[@]}"; do
    MODEL="${MODELS[$model_idx]}"
    MODEL_NAME="${MODEL_NAMES[$model_idx]}"
    LORA_ADAPTER="${LORA_ADAPTERS[$model_idx]}"

    for prompt_idx in "${!PROMPTS[@]}"; do
        PROMPT="${PROMPTS[$prompt_idx]}"
        PROMPT_NAME="${PROMPT_NAMES[$prompt_idx]}"

        for SEED in "${SEEDS[@]}"; do

            echo "=============================================="
            echo "Model: $MODEL_NAME | Prompt: $PROMPT_NAME | Seed: $SEED | LoRA: NO"
            echo "=============================================="
            python gen_text.py \
                --config_file "$CONFIG_FILE" \
                --tokenizer_path "$TOKENIZER_PATH" \
                --checkpoint "$MODEL" \
                --prompt "$PROMPT" \
                --max_new_tokens $MAX_NEW_TOKENS \
                --temperature $TEMPERATURE \
                --repetition_penalty $REPETITION_PENALTY \
                --seed=$SEED

            echo ""
            echo "=============================================="
            echo "Model: $MODEL_NAME | Prompt: $PROMPT_NAME | Seed: $SEED | LoRA: YES"
            echo "=============================================="
            python gen_text.py \
                --config_file "$CONFIG_FILE" \
                --tokenizer_path "$TOKENIZER_PATH" \
                --checkpoint "$MODEL" \
                --lora_adapter "$LORA_ADAPTER" \
                --lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_scope "$LORA_SCOPE" \
                --train_embeddings \
                --prompt "$PROMPT" \
                --max_new_tokens $MAX_NEW_TOKENS \
                --temperature $TEMPERATURE \
                --repetition_penalty $REPETITION_PENALTY \
                --seed=$SEED

            echo ""
        done
    done
done