# 2021-VRDL-HW4

This repository is the official implementation of [2021 VRDL HW4](https://codalab.lisn.upsaclay.fr/competitions/622?secret_key=4e06d660-cd84-429c-971b-79d15f78d400#learn_the_details). 


## Reproducing Submission
1. [Requirements](#Requirements)
2. [Pretrained_Model](#Pretrained_Model)
3. [Data](#Data)
4. [Inference](#Inference)

### Environment
-Google Colab



## Requirements
All requirements is satisfied on Colab.

## Pretrained_Model
Download the [model](https://drive.google.com/file/d/15GLAv1nd9LT2lZbQHNDoA4Yoi_rlu69Q/view?usp=sharing)

and put it in "2021-VRDL-HW3/model/"



## Data
Download the dataset from [here](https://drive.google.com/file/d/1WCOhLfEreUA-2H_J7NmgvN1hefuvEREs/view?usp=sharing)

and unzip it in "2021-VRDL-HW3/".

There will be three folder in it, "train", "test", "stage1_test".


## Train

```Train
python nuclei_train.py --dir_log logs
```



## Inference

```Inference
python samples/nucleus/nucleus.py detect --dataset=dataset --subset=stage1_test --weights=model/mask_rcnn_nuclei_train_0026.h5
```
answer.json will be in "2021 VRDL HW3/"


## Reference
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution
https://github.com/junnjun/SuperResolution_SRResNet
https://github.com/Saafke/EDSR_Tensorflow
