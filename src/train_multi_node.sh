#!/bin/bash
# EDIT ME: job name, queue, and resource selection are specific to your cluster's PBS
# setup -- these are just illustrative example values.
#PBS -N multinode-training
#PBS -q gpu
#PBS -l select=2:ncpus=16:ngpus=8:mpiprocs=1
#PBS -l place=scatter
#PBS -l walltime=04:00:00
#PBS -o src/hpc_logs/
#PBS -e src/hpc_logs/

set -euo pipefail

cd "$PBS_O_WORKDIR"

# EDIT ME: module name/version is specific to your cluster.
module load cuda12.9/toolkit/12.9
source "$PBS_O_WORKDIR/.venv/bin/activate"

echo "===== PBS_NODEFILE ====="
cat "$PBS_NODEFILE"
echo "========================"

# Unique allocated nodes
nodes=($(sort -u "$PBS_NODEFILE"))
NNODES=${#nodes[@]}
MASTER_ADDR=${nodes[0]}
MASTER_PORT=29500

echo "Master node : $MASTER_ADDR"
echo "Num nodes   : $NNODES"

# Every rank on every node runs its own `hydra.main`, so the run directory
# (and therefore the snapshot path, which is derived from it) must be fixed
# once here and passed down explicitly -- letting each process resolve
# Hydra's `now:` timestamp independently risks different nodes disagreeing
# on the directory. Pass run_dir=<existing outputs/... dir> via `qsub -v` to
# resume a previous run instead of starting a fresh one.
EXPERIMENT=${experiment:-example}
RUN_DIR=${run_dir:-"outputs/${EXPERIMENT}/$(date +%Y-%m-%d_%H-%M-%S)"}
echo "Experiment  : $EXPERIMENT"
echo "Run dir     : $RUN_DIR"

# Extra Hydra overrides layered on top of whatever `experiment` selects (epoch count,
# dataset stride, LR, ...), e.g.:
#   qsub -v experiment=example,hydra_overrides="trainer_config.max_epochs=25" src/train_multi_node.sh
HYDRA_OVERRIDES=${hydra_overrides:-}
echo "Overrides   : $HYDRA_OVERRIDES"

# Detect GPUs per node
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
echo "GPUs/node   : $GPUS_PER_NODE"

export LOGLEVEL=INFO
export NCCL_DEBUG=INFO

# Launch one torchrun per node
for ((i=0; i<NNODES; i++)); do
    pbsdsh -n "$i" -- bash -lc "
        set -e

        cd '$PBS_O_WORKDIR'

        module load cuda12.9/toolkit/12.9
        source '$PBS_O_WORKDIR/.venv/bin/activate'

        export LOGLEVEL=INFO
        export NCCL_DEBUG=INFO

        echo =========================
        echo HOST=\$(hostname)
        echo NODE_RANK=\$PBS_NODENUM
        echo MASTER_ADDR=$MASTER_ADDR
        echo MASTER_PORT=$MASTER_PORT
        echo ==========================

        torchrun \
            --nnodes=$NNODES \
            --node_rank=\$PBS_NODENUM \
            --nproc_per_node=$GPUS_PER_NODE \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            src/main.py \
            experiment=$EXPERIMENT \
            hydra.run.dir=$RUN_DIR \
            $HYDRA_OVERRIDES
    " &
done

wait