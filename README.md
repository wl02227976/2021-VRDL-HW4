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
Download the [model](https://drive.google.com/file/d/1fkPmy5dTqZ1wR0ukahXBirZIwNJeAh-E/view?usp=sharing)

and put it in "2021-VRDL-HW4/models/"



## Data
Download the dataset from [here](https://drive.google.com/file/d/1ewI1tdXqpkxRwh06tLYwaGpdBEY5PBzw/view?usp=sharing)

and unzip it in "2021-VRDL-HW4/data/".


## Train

```Train
python train.py
```


## Inference

```Inference
python inference.py
```
The result will be in "2021-VRDL-HW4/output/".


## Reference
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution
https://github.com/junnjun/SuperResolution_SRResNet
https://github.com/Saafke/EDSR_Tensorflow
