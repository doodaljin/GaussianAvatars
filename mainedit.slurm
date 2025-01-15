#!/bin/bash
#SBATCH --job-name=tdd-ga
#SBATCH --output=ga_out.txt
#SBATCH --error=ga_err.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1   
#SBATCH --mem-per-cpu=20G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00      

source miniconda3/bin/activate
conda activate ga 
cd GaussianAvatars
python edit.py \
-s export/UNION10_218_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
-m output/hulk_edit_new_strategy \
-e output/hulk_edit_new_strategy/edit_temp \
-p "Turn him into Hulk" \
--start_checkpoint output/UNION10EMOEXP_218_eval_2/chkpnt600000.pth \
--eval --bind_to_mesh --white_background --port 60000
conda deactivate