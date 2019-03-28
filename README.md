# Fruit Classifier using Pytorch
Fruit classification using Kaggle Dataset [Fruit-360](https://www.kaggle.com/moltean/fruits) in pytorch.
This repository contains some code on :
a) Creation of custom dataset using pytorch. Look at fruit.py to understand how the custom dataset can be prepared from a set of training and test images.
b) Creation of a Network in pytorch which is simplier to create and try out any changes to it.
c) Easy to train and test.

# Training and Validation of Fruit-360 dataset.

## Step 1
The same concept applies to all different kinds of datasets.
Firstly, load all the images that are downloaded from the above link and convert them into npy files.
Advantage of using npy files is to use only 4 files named train_data.npy, train_labels.npy and validation_data.npy , validation_labels.npy
rather than using thousands of files for pre-processing.

To convert your training and validation dataset into npy files use the below script.

```
python load_dataset.py --dataset-dir <Dataset Path>
```

This creates train_data.npy, train_labels.npy, validation_data.npy, validation_labels.npy

## Step 2
Use the dataset files that are created above, train the fruit classifier and evaluate the model.

```
python train.py --data-dir <npy files folder> [--epochs <default:10>]
```

This generates a log that trains the network for each epoch and finally do inference against the validation dataset spits out the validation accuracy. 



