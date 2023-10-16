# SparseMA Algorithm
This repository contains code to reproduce results from the paper:

Sparse Multimodal Attack for Vision-Language Adversarial Example Generation

## Requirements

- torch==1.7.0
- torchvision==0.8.0
- transformers==2.3.0
- nltk == 3.7
- boto3
- scikit-learn
- hnswlib
- pandas
- pyarrow == 10.0.1
- language_tool_python
- pyyaml==5.4.1
- scikit-image
- tensorflow_gpu == 2.11.0
- tensorflow-hub
- timm
- opencv-python

## Datesets

There are three datasets used in our experiments including [MVSA-Single](https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/), [MVSA-Multi](https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/), [CrisisMMD](https://crisisnlp.qcri.org/crisismmd#data_version2.0). You could put these datasets into the directory `./data/dataset/MVSA_Single`, `./data/dataset/MVSA`, `./data/dataset/CrisisMMD_v2.0`,  respectively. Then, you could run `mvsa_preprocess.py` to process the `MVSA_Single` and `MVSA-Multi` datasets, and `crisismmd_preprocess.py` to process the `CrisisMMD` dataset.

   ```shell
python dataloader/mvsa_preprocess.py --mvsa_single_data_path ./data/dataset/MVSA_Single --mvsa_multiple_data_path ./data/dataset/MVSA

python dataloader/crisismmd_preprocess.py --dataset_filename ./data/dataset/CrisisMMD_v2.0
   ```

## Target Model

We adopt three multimodal models to predict the label, including `CLIP_ViT`, `CLIP_Res` and `ALBEF`. Then you could run `./scripts/train/mvsa_single/clip_fusion.sh` to train the `CLIP_ViT` model using the `MVSA_Single` dataset. The trained model would be placed under the `./data/model/` folder.

   ```shell
git lfs install
git clone https://huggingface.co/bert-base-uncased
   ```


## Dependencies

There are three dependencies for this project adopted from the [github repo](https://github.com/RishabhMaheshwary/hard-label-attack/tree/main/data) of [HLBB](https://arxiv.org/abs/2012.14956). Download and put `glove.6B.200d.txt` to the directory `./data/embedding`. And put `counter-fitted-vectors.txt` and the top synonym file `mat.txt` to the directory `./data/aux_files`.

- [GloVe vecors](https://nlp.stanford.edu/data/glove.6B.zip)
- [Counter fitted vectors](https://drive.google.com/file/d/1bayGomljWb6HeYDMTDKXrh0HackKtSlx/view)
- [Mat](https://drive.google.com/file/d/1AIz8Imvv8OmHxVwY5kx10iwKAUzD6ODx/view)


## File Description

- `train_classifier.py`: Train the victim model.
- `attack.py`: Attack the target model for vision-languange classification with various attacks.
- `config.py`: Parameters of attack for all datasets.
- `./adv_method`: Implementation for our SparseMA.
- `./data`: Dataset, embedding matrix and various aux files.
- `./model_loader`: Target model, including CNN_VGG_Fusion, LSTM_ResNet_Fusion and BERT_ViT_Fusion.
- `./utils`: Helper functions for building dictionaries, loading data, and processing embedding matrix etc.
- `./parameter`: All hyper-parameters of our SparseMA for various target models and datasets in our main experiments.
- `./scripts`: Commands to run the attack.


## Experiments

Taking the SparseMA attack on CLIP_ViT using MVSA_Single dataset for example, you could run the following command:

   ```shell
sh scripts/sparse_ma/clip_vit_mvsa_single.sh
   ```

You could change the hyper-parameters of SparseMA in the `./parameter/sparse_ma/clip_vit_mvsa_single.yaml` if necessary.


## Citation

If you find this code and data useful, please consider citing the original work by authors:



