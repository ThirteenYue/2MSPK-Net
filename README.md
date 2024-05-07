# 2MSPK-Net
This repo is the official implementation of
['2MSPK-Net: A Nuclei Segmentation Network Based on Multi-Scale, Multi-Dimensional Attention, and SAM Prior Knowledge']
<p align="center">
  <img src="https://github.com/ThirteenYue/2MSPK-Net/blob/master/prior.png" width="50%" height="50%" />
</p>
We proposed a segmentation method based on SAM prior knowledge guidance strategy, and the above is a schematic diagram of integrating SAM prior knowledge.For detailed method introduction, please read the original article

## Requirements
Install from the ```requirements.txt``` using:
```angular2html
pip install -r requirements.txt
```

## Usage

*Note: If you have some problems with the code, the [issues](https://github.com/ThirteenYue/2MSPK-Net/issues) may help.*

### 1. Data Preparation
#### 1.1. GlaS and MoNuSeg Datasets
🔥 The original data can be downloaded in following links:
* MoNuSeg Dataset - [Link (Original)](https://monuseg.grand-challenge.org/Data/)
* TNBC Dataset - [Link (Original)](https://paperswithcode.com/dataset/tnbc)

Then prepare the datasets in the following format for easy use of the code:
```angular2html
├── Dataset
    ├── MoNusg
    │   ├── test
    │   │   ├── boundary_priors
    │   │   ├── images
    │   │   ├── masks
    │   │   └── seg_priors
    │   ├── train
    │   │   ├── boundary_priors
    │   │   ├── images
    │   │   ├── masks
    │   │   └── seg_priors	
    │   └── val
    │   │   ├── boundary_priors
    │   │   ├── images
    │   │   ├── masks
            └── seg_priors
```

### 2. Training
During the training process, the data were uniformly resized to 256×256 pixels and data augmentation was applied, including affine transformation, random flipping, and random rotation. Gradient descent was performed using the Adam optimizer with \beta_1 set to 0.9 and \beta_2 set to 0.999. The initial learning rate was set to 1\times{10}^{-4}, and an adaptive learning rate decay strategy was employed. If the loss on the validation set did not decrease after every 20 epochs, the learning rate was reduced by a factor of 0.5. The batch size was set to 4, and the training was completed after 600 epochs. 


#### 2.2 Pre-training
We didn't use any pre-trained weights


### 3. Testing
We also added SAM prior area maps and contour maps to the test data set
*You can generate area maps and contour mapsarea and contour plots using SAM_prior.py
We will announce other test and visualization codes later

## Reference
* UNet:https://github.com/LeeJunHyun/Image_Segmentation#u-net
* UNet++: https://github.com/qubvel/segmentation_models.pytorch
* https://github.com/huangmozhilv/u2net_torch
* Attention U-Net: https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets
* TransUNet: https://github.com/Beckschen/TransUNet
* Swin-Unet: https://github.com/HuCaoFighting/Swin-Unet


## Citations


If this code is helpful for your study, please cite:
```
```


## Contact 
Gongtao Yue([thirteen_yue@163.com](thirteen_yue@163.com))
