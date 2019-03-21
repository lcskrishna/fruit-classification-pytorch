import os
import sys
import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser();
parser.add_argument('--train-data', type=str, required=True, help="Path to the training npy")
parser.add_argument('--original-image', type=str, required=True, help="Path to the original image")

args = parser.parse_args()

train_data_npy = os.path.abspath(os.path.join(args.train_data, "train_data.npy"))
train_labels_npy = os.path.abspath(os.path.join(args.train_data, "train_labels.npy"))

print ("INFO: Training image path is : {}".format(train_data_npy))
print ("INFO: Training image labels path is : {}".format(train_labels_npy))

train_data = np.load(train_data_npy)
train_labels = np.load(train_labels_npy)

print ("Shape of training data is : {}".format(train_data.shape))
print ("Shape of training labels is : {}".format(train_labels.shape))

first_image = train_data[1, :, :, :]
print ("First image shape is : {}".format(first_image.shape))
dims = first_image.shape
first_image_new = np.reshape(first_image, (dims[2], dims[1], dims[0]))
print ("Formatted first image shape is : {}".format(first_image_new.shape))

## Write out the image.
cv2.imwrite('test_image.png', first_image_new)
print ("INFO: The image is written to a file (test_image.png)")

## Read both the images.
original_image = cv2.imread(args.original_image, cv2.IMREAD_COLOR)
test_image = cv2.imread('test_image.png', cv2.IMREAD_COLOR)

difference = cv2.subtract(original_image, test_image)
result = not np.any(difference)

if result is True:
    print ("Pictures are same.")
else:
    print ("Pictures are not same.")
    max_val = np.max(difference)
    print (max_val)
