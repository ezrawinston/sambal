#!/bin/bash

# This is an example script for running the BERT model pretraining on a single node with 8 GPUs.
# You'll most likely have to adjust the script to match your setup.

# This script is called from the slurm.sh script, which sets up the environment and calls this script on each GPU

# Launch script used by slurm scripts, don't invoke directly.
source /opt/miniconda3/bin/activate pytorch

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export OMP_NUM_THREADS=1

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
echo "Launching on $SLURMD_NODENAME ($SLURM_PROCID/$SLURM_JOB_NUM_NODES)," \
     "master $MASTER_ADDR port $MASTER_PORT," \
     "GPUs $SLURM_GPUS_ON_NODE," \
     "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

python -u "$@"
