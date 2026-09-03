TODO: Create single GPU training


# Hybrid BERT Pre-training

This repository contains code for pre-training BERT models using a hybrid approach that combines masked language modeling (MLM) and causal language modeling (CLM) objectives. The implementation supports distributed training across multiple GPUs using PyTorch's DistributedDataParallel.

## Features

- Hybrid training combining MLM and CLM objectives with configurable ratios
- Distributed training support with SLURM integration
- Mixed precision training with bfloat16
- Gradient accumulation and clipping
- Exponential Moving Average (EMA) model maintenance
- Dynamic sequence length and batch size scheduling
- Cosine learning rate schedule with warmup and cooldown
- Support for LAMB and AdamW optimizers
- Wandb integration for experiment tracking
- Token-weighted loss computation option
- Comprehensive validation metrics

## Training Script

The main training script supports numerous configuration options. Here's how to use it for multi-gpu:

```bash
python train_multi_gpu.py \
    --train_path="PATH_TO_TRAINING_DATA" \
    --valid_path="PATH_TO_VALIDATION_DATA" \
    --config_file="PATH_TO_MODEL_CONFIG" \
    --tokenizer_path="PATH_TO_TOKENIZER" \
    --output_dir="PATH_TO_SAVE_CHECKPOINTS" \
    # WandB information + model name
    --name="RUN_NAME" \
    --wandb_project="YOUR_WANDB_PROJECT_NAME" \
    --wandb_entity="YOUR_WANDB_ENTITY" \
    # Training Configuration
    --hybrid_numerator=15 \
    --hybrid_denominator=16 \
    --seq_length=128 \
    --local_batch_size=128 \
    --global_batch_size=32768 \
    --learning_rate=1e-2 \
    --max_steps=15625 \
    # Optimizer & Scheduler
    --optimizer="lamb" \
    --weight_decay=0.1 \
    --warmup_proportion=0.016 \
    --cooldown_proportion=0.016 \
    # Masking Configuration
    --mask_p_start=0.3 \
    --mask_p_end=0.15 \
    --mask_random_p=0.1 \
    --mask_keep_p=0.1 \
    # Training Control
    --mixed_precision \
    --validate_every=1000 \
    --save_every=1000 \
    --seed=42
```

And for single-gpu:

```bash
python train_single_gpu.py \
    --train_path="PATH_TO_TRAINING_DATA" \
    --valid_path="PATH_TO_VALIDATION_DATA" \
    --config_file="PATH_TO_MODEL_CONFIG" \
    --tokenizer_path="PATH_TO_TOKENIZER" \
    --output_dir="PATH_TO_SAVE_CHECKPOINTS" \
    # WandB information + model name
    --name="RUN_NAME" \
    --wandb_project="YOUR_WANDB_PROJECT_NAME" \
    --wandb_entity="YOUR_WANDB_ENTITY" \
    # Training Configuration
    --hybrid_numerator=15 \
    --hybrid_denominator=16 \
    --seq_length=128 \
    --local_batch_size=128 \
    --global_batch_size=32768 \
    --learning_rate=1e-2 \
    --max_steps=15625 \
    # Optimizer & Scheduler
    --optimizer="lamb" \
    --weight_decay=0.1 \
    --warmup_proportion=0.016 \
    --cooldown_proportion=0.016 \
    # Masking Configuration
    --mask_p_start=0.3 \
    --mask_p_end=0.15 \
    --mask_random_p=0.1 \
    --mask_keep_p=0.1 \
    # Training Control
    --mixed_precision \
    --validate_every=1000 \
    --save_every=1000 \
    --seed=42
```

## Key Components

### Hybrid Training

The code implements a hybrid training approach where:
- A fraction of GPUs (controlled by `hybrid_numerator/hybrid_denominator`) trains with MLM
- The remaining GPUs train with CLM
- Losses are automatically weighted and combined based on the ratio

Important note: The code that assumes multi-gpu needs the number of GPUs being used to be a multiple of your `hybrid_denominator` i.e. if you want to train with a 1:3 causal-to-mask ratio, you have to have a multiple of 4 number of GPUs.

### Dynamic Scheduling

The training includes several dynamic scheduling features:
- Sequence length increases during training (128→256→512)
- Batch size scales linearly from reduced to full size
- Learning rate follows a cosine schedule with warmup and cooldown

### Model Architecture

The model follows the BERT architecture with:
- Support for variable sequence lengths
- Configurable model dimensions via config file
- Optional token-weighted loss computation
- Z-loss regularization option
- `model.py` contains the base LTG-BERT architecture model
- `model_extra.py` contains the model architecture from the paper

### Validation and Monitoring

Comprehensive metrics are tracked including:
- Training and validation loss/perplexity
- Token prediction accuracy
- Gradient norms
- Learning rates
- Batch sizes and sequence lengths
- Masking probabilities

## Usage Notes

### Environment Setup

The code requires:
- PyTorch with CUDA support
- SLURM for distributed training
- Wandb for experiment tracking (The code is quite intertwined with WandB integration)
- HuggingFace Tokenizers

### Data Preparation

Training data should be preprocessed and tokenized before training. The code expects:
- Binary format for efficient loading
- Pre-computed token distributions for masked prediction
- Separate training and validation sets

### Distributed Training

The code is designed for multi-GPU training and:
- Automatically handles distributed setup via SLURM
- Supports gradient accumulation for large batch sizes
- Implements efficient mixed-precision training
- Uses DDP for model parallelization

### Checkpointing

The training script saves:
- Regular model weights
- EMA model weights
- Full training state (for resumption)
- Optimizer and scheduler states

### Monitoring

Training progress is logged to Wandb with:
- Separate tracking for MLM and CLM losses
- Validation metrics every N steps (Validation is done in an MNTP fashion and not a causal fashion)
- Resource utilization statistics
- Hyperparameter tracking

## Customization

The code supports several customization points:
- Custom model architectures via config files
- Adjustable training objectives and ratios
- Flexible optimization schemes
- Custom tokenization approaches

## Common Issues

1. **Memory Management**:
   - Reduce the local batch size if OOM occurs
   - Enable mixed precision training
   - Utilize gradient accumulation

2. **Distributed Training**:
   - Ensure SLURM environment variables are set
   - Check GPU visibility and allocation
   - Monitor process ranks and world size

3. **Data Loading**:
   - Use appropriate number of workers
   - Enable pin memory for GPU training
   - Handle dataset sharding properly

4. **Convergence Issues**:
   - Adjust learning rate and warmup
   - Monitor gradient norms
   - Check loss scaling and weighting