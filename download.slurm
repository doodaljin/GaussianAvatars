#!/bin/bash
#SBATCH --job-name=tdd-dljob
#SBATCH --output=dlout.txt
#SBATCH --error=dlerr.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1   
#SBATCH --mem-per-cpu=4G     
#SBATCH --time=6:00:00 

source miniconda3/bin/activate
conda activate testenv
pip install gdown
cd GaussianAvatars/data 
gdown https://drive.google.com/uc?id=1-Dau4YyGGKcW5H-vLMFCQ8WvJQYfBcH_ 
unzip 218.zip
conda deactivate