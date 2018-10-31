#!/bin/bash
#
#SBATCH --job-name=TinyNet
#SBATCH --output=./out_files/%u_%j-%A.out
#
#SBATCH --ntasks=1
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --time=03:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hos002

source /gpfs/work/hos00/hos002/l1_setup.sh
srun python src/main.py -i /gpfs/work/hos00/hos002/datasets/cifar10/ -o ./results/ -j ${SLURM_JOB_ID} -a 5 -d 10 -n 1 -c 40 -g True -s $1 -b $2 -l $3 -p True

