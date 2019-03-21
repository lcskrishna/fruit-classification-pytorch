import argparse
import os
import sys
import numpy as np
import cv2
import glob

print ("INFO: all the modules are imported.")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help='Path to the dataset folder')

args = parser.parse_args()

fruit_names = [
    'AppleBraeburn',
    'AppleGolden1',
    'AppleGolden2',
    'AppleGolden3',
    'AppleGrannySmith',
    'AppleRed1',
    'AppleRed2',
    'AppleRed3',
    'AppleRedDelicious',
    'AppleRedYellow1',
    'AppleRedYellow2',
    'Apricot',
    'Avocado',
    'Avocado_ripe',
    'Banana',
    'Banana_Lady_Finger',
    'Banana_Red',
    'Cactus_fruit',
    'Cantaloupe1',
    'Cantaloupe2',
    'Carambula',
    'Cherry1',
    'Cherry2',
    'CherryRainier',
    'CherryWaxBlack',
    'CherryWaxRed',
    'CherryWaxYellow',
    'Chestnut',
    'Clementine',
    'Cocos',
    'Dates',
    'Granadilla',
    'GrapeBlue',
    'GrapefruitPink',
    'GrapefruitWhite',
    'GrapePink',
    'GrapeWhite',
    'GrapeWhite2',
    'GrapeWhite3',
    'GrapeWhite4',
    'Guava',
    'Hazelnut',
    'Huckleberry',
    'Kaki',
    'Kiwi',
    'Kumquats',
    'Lemon',
    'LemonMeyer',
    'Limes',
    'Lychee',
    'Mandarine',
    'Mango',
    'Mangostan',
    'Maracuja',
    'Melon_Piel_de_Sapo',
    'Mulberry',
    'Nectarine',
    'Orange',
    'Papaya',
    'PassionFruit',
    'Peach',
    'Peach2',
    'PeachFlat',
    'Pear',
    'PearAbate',
    'PearKaiser',
    'PearMonster',
    'PearWilliams',
    'Pepino',
    'Physalis',
    'Physalis_with_Husk',
    'Pineapple',
    'PineappleMini',
    'PitahayaRed',
    'Plum',
    'Plum2',
    'Plum3',
    'Pomegranate',
    'PomeloSweetie',
    'Quince',
    'Rambutan',
    'Raspberry',
    'Salak',
    'Strawberry',
    'Strawberry_Wedge',
    'Tamarillo',
    'Tangelo',
    'Tomato1',
    'Tomato2',
    'Tomato3',
    'Tomato4',
    'TomatoCherryRed',
    'TomatoMaroon',
    'Walnut'
]

image_path = args.dataset
print ("INFO: Training image path is : {}".format(image_path))

## Creation of training data.
train_data = []
train_labels = []
for fruit in fruit_names:
    print (fruit)
    folder_path = os.path.join(image_path, "Training", fruit)
    images = os.listdir(folder_path)

    for i in range(len(images)):
        final_path = os.path.join(folder_path, images[i])
        img =  cv2.imread(final_path, cv2.IMREAD_COLOR)
        dims = np.shape(img)
        img = np.reshape(img, (dims[2], dims[0], dims[1]))
        train_data.append(img)
        train_labels.append(fruit_names.index(fruit))

train_data = np.array(train_data)
print (train_data.shape)
train_labels = np.array(train_labels)
print (train_labels.shape)

print ("OK: Training data created.")


### saving the data into a file.
np.save('train_data.npy', train_data)
check = np.load('train_data.npy')
np.save('train_labels.npy', train_labels)
check2 = np.load('train_labels.npy')

print (check.shape)
print (check2.shape)


validation_data = []
validation_labels = []
for fruit in fruit_names:
    print (fruit)
    folder_path = os.path.join(image_path, "Test", fruit)
    images = os.listdir(folder_path)
    
    for i in range(len(images)):
        final_path = os.path.join(folder_path, images[i])
        if not os.path.isfile(final_path):
            print ("This path doeesn't exist : {}".format(final_path))
            continue
        img = cv2.imread(final_path, cv2.IMREAD_COLOR)
        dims = np.shape(img)
        img = np.reshape(img, (dims[2], dims[0], dims[1]))
        validation_data.append(img)
        validation_labels.append(fruit_names.index(fruit))

validation_data = np.array(validation_data)
print (validation_data.shape)
validation_labels = np.array(validation_labels)
print (validation_labels.shape)

print ("OK: Validation data created.")

### saving the data into a file.
np.save('validation_data.npy', validation_data)
check = np.load('validation_data.npy')
np.save('validation_labels.npy', validation_labels)
check2 = np.load('validation_labels.npy')
#
print (check.shape)
print (check2.shape)

print (len(fruit_names))
