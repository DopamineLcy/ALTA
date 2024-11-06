# ALTA
This is the supplementary material that includes the code for the paper: Efficient medical vision-language alignment through adapting masked vision models.

Some code of this repository is borrowed from [MAE](https://github.com/facebookresearch/mae), [MRM](https://github.com/RL4M/MRM-pytorch), [AIM](https://adapt-image-models.github.io/) and [huggingface](https://huggingface.co).

The code will be publicly available after the reviewing process.

## Getting started
### 1 Requirement
OS: Ubuntu 20.04 LTS.

Language: Python 3.10.8

If you are using conda, we provide an easy way to continue:

      conda env create -f environment.yaml
      pip install -r requirements.txt


### 2 Data preparation
- We use MIMIC-CXR-JPG for pre-training. You can acquire more information about this dataset at [Johnson et al. MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).
- The dataset directory specified in run.sh includes the MIMIC-CXR-JPG dataset and you need to prepare files "train.csv" and "valid.csv", then put them into the dateset directory [MIMIC-CXR_dataset](MIMIC-CXR_dataset).
- The file "training.csv" includes many columns for each line, including: image_path, auxview_image_path, last_image_path, last_auxview_image_path, report, which stands for the path of current frontal image, current lateral image, prior frontal image, prior lateral image, and the content of report, respectively.
- Besides, (RSNA Pneumonia) is used for validation, please put "val_list.txt' into the directory of [RSNA_dataset](RSNA_dataset). The dataset can be downloaded from https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge

### 3 Pre-train models preparation

- Get pre-trained weights of [MRM](https://github.com/RL4M/MRM-pytorch) and put the file into [vision_encoder_weights](vision_encoder_weights).

- Get pre-trained language model from [BiomedVLP-CXR-BERT-specialized](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized) and put the files into the current directory.


### 4 Start Pre-training

- Set the data path, GPU IDs, batch size, output directory, and other parameters in [run.sh](run.sh).

- Start training by running

      chmod a+x run.sh
      ./run.sh