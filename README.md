# fruit-classification-pytorch
Fruit classification using Kaggle Dataset [Fruit-360](https://www.kaggle.com/moltean/fruits) in pytorch

## How to run the test
Make sure to download the directory in a folder.

First load the datasets into npy files.

```
python load_dataset.py --dataset-dir <Dataset Path>

```
This creates train_data.npy, train_labels.npy, validation_data.npy, validation_labels.npy

Using this, train and test the fruit dataset using train.py

```
python train.py --data-dir <npy files folder>

```


