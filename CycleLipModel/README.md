# LipCycle Model 

An integration made using the lip2wav-multispeaker (Pytorch implementation) and the Wav2lip model
- The Files required to install are the same as that of the lip2wav and wav2lip models

Lip2wav - multispeaker (https://github.com/Rudrabha/Lip2Wav/tree/multispeaker)



## Preprocessing
Note for preprocessing the only code to be preprocessed should be that of the lip2wav-multispeaker-original as wav2lip shares the same preprocessed data.

## Training
- In order to perform training execute the following function
    
- Note that before training begins you must have pretrained a lip2wav model as well as a wav2lip model. Refer to the relevant sections before training cyclelip

python cyclelip_train.py --data_root <dir/to/preprocessed/data> \
--checkpoint_dir_wav2lip <dir/to/save/wav2lip/checkpoint/folder> \
--syncnet_checkpoint_path <dir/to/load/syncnet/checkpoint/folder> \
--checkpoint_path_wav2lip  <dir/to/load/wav2lip/checkpoint/folder> \
--disc_checkpoint_path <dir/to/load/wav2lip/disc/checkpoint/folder> \
--checkpoint_dir_lip2wav <dir/to/save/lip2wav/checkpoint/folder> \
--checkpoint_path_lip2wav <dir/to/load/lip2wav/checkpoint/folder>
--checkpoint_dir_loss_data <dir/to/loss/data> 





## Inference and evaluation
- For inference refer back to the lip2wav-pytorch folder and wav2lip folders


## Pretrained Model
Pretrained models can be found under the model folder under the main parent folder

## potential areas of improvement
- Improve pytorch conversion of lip2wav model
- Relook at lost code utilized


