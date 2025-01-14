#!/bin/bash
#SBATCH --job-name=ltnghia11-job
#SBATCH --output=out.txt
#SBATCH --error=err.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1   
#SBATCH --mem-per-cpu=8G     
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00      

source miniconda3/bin/activate
conda activate ga 
conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit ninja -y
ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"
conda env config vars set CUDA_HOME=$CONDA_PREFIX
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
cd GaussianAvatars
pip install -r requirements2.txt
pip install "jax[cuda12_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda deactivate