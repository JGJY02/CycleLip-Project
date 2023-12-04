#!/bin/bash
#SBATCH --job-name=tensorflow_lip2wav
#SBATCH --account=za99
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=m3g

source /scratch/za99/jgoh/miniconda/bin/activate
conda activate lipwav
cd /fs04/za99/jgoh/Lip2Wav-multispeaker
python train.py base_tensorflow_model_sample --data_root /fs04/za99/scratch_nobackup/jgoh/Preprocessed_VoxCeleb_Lip2Wav --checkpoint_interval 5000
