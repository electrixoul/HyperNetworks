#!/bin/bash

# Run HyperNetworks GPU training in the sae_eeg conda environment
# This script can be used to train the model with customizable parameters

# Default parameters
BATCH_SIZE=128
EPOCHS=200
LEARNING_RATE=0.002
WEIGHT_DECAY=0.0005
CHECKPOINT_PATH="./hypernetworks_cifar_gpu.pth"
RESUME=false
USE_WANDB=true
WANDB_PROJECT="hypernetworks-gpu"
WANDB_NAME=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --batch_size)
      BATCH_SIZE="$2"
      shift
      shift
      ;;
    --epochs)
      EPOCHS="$2"
      shift
      shift
      ;;
    --lr)
      LEARNING_RATE="$2"
      shift
      shift
      ;;
    --weight_decay)
      WEIGHT_DECAY="$2"
      shift
      shift
      ;;
    --checkpoint_path)
      CHECKPOINT_PATH="$2"
      shift
      shift
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    --no_wandb)
      USE_WANDB=false
      shift
      ;;
    --wandb_project)
      WANDB_PROJECT="$2"
      shift
      shift
      ;;
    --wandb_name)
      WANDB_NAME="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Print configuration
echo "=== HyperNetworks GPU Training ==="
echo "Using conda environment: sae_eeg"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Weight decay: $WEIGHT_DECAY"
echo "Checkpoint path: $CHECKPOINT_PATH"
if [ "$RESUME" = true ]; then
  echo "Resuming from checkpoint: Yes"
  RESUME_FLAG="--resume"
else
  echo "Resuming from checkpoint: No"
  RESUME_FLAG=""
fi

echo "WandB logging: $([ "$USE_WANDB" = true ] && echo "Enabled" || echo "Disabled")"
if [ "$USE_WANDB" = true ]; then
  echo "WandB project: $WANDB_PROJECT"
  if [ -n "$WANDB_NAME" ]; then
    echo "WandB run name: $WANDB_NAME"
    WANDB_NAME_FLAG="--wandb_name $WANDB_NAME"
  else
    echo "WandB run name: Auto-generated"
    WANDB_NAME_FLAG=""
  fi
  WANDB_FLAGS="--wandb_project $WANDB_PROJECT $WANDB_NAME_FLAG"
else
  WANDB_FLAGS="--no_wandb"
fi

echo -e "\nStarting training...\n"

# Run the training script in the sae_eeg conda environment
cd HyperNetworks_GPU && conda run -n sae_eeg python train.py \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --lr $LEARNING_RATE \
  --weight_decay $WEIGHT_DECAY \
  --checkpoint_path $CHECKPOINT_PATH \
  $RESUME_FLAG \
  $WANDB_FLAGS
