#!/bin/bash  
  
# Usage: ./run.sh train_robotics/assembly/qwen2_5vl_full_sft_assembly_single_turn_conv.yaml

# Check if the configuration file path argument is provided  
if [ -z "$1" ]; then  
  echo "Usage: $0 <config-file-path>"  
  exit 1  
fi  

CONFIG_FILE_PATH=$1  

# export WANDB_RESUME=allow
# export WANDB_RUN_ID=gvuydzqj
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export DEBUG_MODE="true"
export LOG_PATH="./TW_full_sft.txt"

huggingface-cli login --token "${HF_TOKEN:?Please export HF_TOKEN before running this script}"


FORCE_TORCHRUN=1 llamafactory-cli train $CONFIG_FILE_PATH
