# FCN

This code provides Pytorch implementation of FCN architecture on synthetic dataset and real world dataset from scratch as well as finetuning.
There are some custom utils functions for joint transformation, evaluation, and average meter.

## Requirements

Tested on Python 3.6.x and Keras 2.3.0 with TF backend version 1.14.0.
* Numpy 
* Torchvision 
* PyTorch 
* Matplotlib
* Pillow

## Codes running

* Install the required dependencies:
 ```javascript
 pip install -r requirements.txt
 ```
 
 ## Description
 `fcn.py`: Model architecture
 `train_games.py`: Train on games dataset
 `train_cityscapes.py`: Train on cityscapes dataset
 `ft_cityscapes.py`: Finetuning on cityscapes
 `eval_cityscapes.py`: Evaluate on test dataset of cityscapes
 `cityscapes.py`: create dataloader for cityscapes dataset
 `games_data.py`: create dataloader for games dataset
 
 To train on games from scratch
 ```
 python train_games.py 

 ```
 To train on cityscapes from scratch
  ```
 python train_cityscapes.py 

 ```
 To train on cityscapes for finetuning
  ```
 python ft_cityscapes.py 

 ```
Evaluate on cityscapes 
```
 python eval_cityscapes.py 

 ```
