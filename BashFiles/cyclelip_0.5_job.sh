#!/bin/bash
#SBATCH --job-name=CycleLipJob_0.5_cycle_weight
#SBATCH --account=za99
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=m3g
#SBATCH --constraint=V100-32G

source /scratch/za99/jgoh/miniconda/bin/activate
conda activate lipwav
cd /fs04/za99/jgoh/Lip2Wav-Wav2Lip-FullModel/CycleModel
echo "Running Cycle Lip model"
python cyclelip_train.py --checkpoint_dir_wav2lip /fs04/za99/jgoh/Lip2Wav-Wav2Lip-FullModel/Models/cyclelip-0.5-weight-model/wav2lip --checkpoint_dir_lip2wav /fs04/za99/jgoh/Lip2Wav-Wav2Lip-FullModel/Models/cyclelip-0.5-weight-model/lip2wav --checkpoint_dir_loss_data /fs04/za99/jgoh/Lip2Wav-Wav2Lip-FullModel/Models/cyclelip-0.5-weight-model/loss_data --cycle_weight 0.5 --checkpoint_path_wav2lip /fs04/za99/jgoh/Lip2Wav-Wav2Lip-FullModel/Models/cyclelip-0.5-weight-model/wav2lip/wav2lip_checkpoint_step000160000.pth --disc_checkpoint_path /fs04/za99/jgoh/Lip2Wav-Wav2Lip-FullModel/Models/cyclelip-0.5-weight-model/wav2lip/disc_checkpoint_step000160000.pth --checkpoint_path_lip2wav /fs04/za99/jgoh/Lip2Wav-Wav2Lip-FullModel/Models/cyclelip-0.5-weight-model/lip2wav/ckpt_160000