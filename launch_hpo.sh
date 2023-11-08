#!/bin/bash
#SBATCH --job-name="hpo_Transformer.py"
#SBATCH --output="hpo.%j.%N.out"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --account=TG-MDE220007
#SBATCH --mem=32G
#SBATCH -t 48:00:00

module purge
module load slurm
module load gpu/0.15.4  gcc/7.2.0
module load pgi
module load cuda/11.0.2

#Environment
source /home/aiacovelli/gromvenv/bin/activate

export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1

#Run the job
python3 hpo_Transformer.py
