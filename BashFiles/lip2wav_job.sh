#!/bin/bash
#SBATCH --job-name=lip2wav
#SBATCH --account=za99
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=m3g

source /scratch/za99/jgoh/miniconda/bin/activate
conda activate lipwav
cd /fs04/za99/jgoh/Lip2Wav-Wav2Lip-FullModel/Lip2Wav-pytorchConversion/LIP2WAV-Converted
echo "Running Cycle Lip model"
python train.py --data_root /fs04/za99/scratch_nobackup/jgoh/Preprocessed_VoxCeleb_Lip2Wav --checkpoint_dir /fs04/za99/jgoh/Lip2Wav-Wav2Lip-FullModel/Lip2Wav-pytorchConversion/LIP2WAV-Converted/training_results/model
