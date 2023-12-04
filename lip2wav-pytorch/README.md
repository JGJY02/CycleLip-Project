# Lip2wav-PyTorch

A conversion made based on the original multispeaker lip2wav tensorflow implementation using the pytorch implementation of the tactron2 as a baseline.

Lip2wav - multispeaker (https://github.com/Rudrabha/Lip2Wav/tree/multispeaker)
tacotron2-pytorch (https://github.com/BogiHsu/Tacotron2-PyTorch)

for the requirements to run use the original lip2wav-multispeaker paper requirements

## Preprocessing
Data to be preprocessed should use the preprocessing codes found in the lip2wav-multispeaker-tensorflow folder. Currently preprocessing is edited to accept text files but the original code also be used to preprocess

python preprocess.py --files_to_process <text/file/of/data/to/preprocess> --data_root <dir/to/raw/files> --preprocessed_root <dir/to/preprocessed/data> --split <data/type>

Once completed you cna then run to obtain the speech embeddings of the speakers

python preprocess_speakers.py --preprocessed_root <dir/to/preprocessed/data>

## Training
- In order to perform training execute the following function
python train.py --data_root <dir/to/preprocessed/data> --checkpoint_dir <dir/to/checkpoint/folder>


## Inference
- For synthesizing wav files, run the following command.
python inference.py --data_root <dir/to/preprocessed/data> --results_root <dir/to/generated/audios> --checkpoint <dir/to/checkpoint/folder>

## Evaluation
- For Scoring on the basis of STOI, ESTOI and PESQ
python score.py -r <dir/to/generated/audios>


## Pretrained Model
Pretrained models can be found under the model folder under the main parent folder


