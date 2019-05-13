#!/usr/bin/env python
# coding: utf-8

import os
import glob
import tqdm
import numpy as np 
import pandas as pd
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt

from preprocess import load_dicom_volume, load_label
from preprocess import resample, make_lungmask, hounsfield_unit_tranformation, window_normalization

from plotting import plot_hu_distribution

train_image_folder = "./input/train-images/image/"
train_label_folder = "./input/train-labels/label/"

train_list = os.listdir(train_image_folder)[:]

bad_train = [
    '.DS_Store',
    '2w2suqniwvkbxnfdpi2qayhlyfouhozt',
    '4rgpktehnvslfprnaatyi0rv75peac7o',
    '6akhpnzfnk9cgkik8k7izk6akx0kbmxp',
    '081j5f5m21t7nb1yd03uncoofau9cfzu',
    'nx5t62d6hzvm1egydllfmg8mti2omeil',
    'oa7nilk4nuzck787quj2yms05hz6k58s',
    'prmp9uolz3qje3x067gb6ze096afmj3e',
    'pvy9t4lp7wxyn1sn813xrqxe5e2c2l4u',
    'qrtx68af7g9q6hbq4g2p66itb31ym3c3',
    's6lyih6jfv9izt34lkah9ywva5wgeqxk',
    'uz15d5odukrbslyuzel6qhgo28pvf5kt',
    'w76zb988p62bqzxi6fprbyd7dju0l58h',
    'hmvsa0loxh3ek2y8rzmcyb6zrrh9mwyp', 
]

# Ignore this data
# if 'hmvsa0loxh3ek2y8rzmcyb6zrrh9mwyp' in train_list:
#     train_list.remove('hmvsa0loxh3ek2y8rzmcyb6zrrh9mwyp')

for s in bad_train:
    if s in train_list:
        train_list.remove(s)
    
print('Train data:', len(train_list))


train_image_npz_folder = './npz-norm/train_images/'
train_label_npz_folder = './npz-norm/train_labels/'

if not os.path.exists(train_image_npz_folder):
    os.makedirs(train_image_npz_folder)

if not os.path.exists(train_label_npz_folder):
    os.makedirs(train_label_npz_folder)
    

for encode in tqdm.tqdm(train_list):
    _, volume_image, info_dict = load_dicom_volume(os.path.join(train_image_folder, encode))
    _, label_array = load_label(os.path.join(train_label_folder, encode + '.nii.gz'), transpose=True)
    
    # Transfer to Hounsfield units (HU)
    volume_image = hounsfield_unit_tranformation(volume_image,
                                         slope=info_dict['RescaleSlope'],
                                         intercept=info_dict['RescaleIntercept'])
    
    # normalization lower_th=-1000., upper_th=400.
    volume_image = window_normalization(volume_image, lower_th=-1000., upper_th=400)
    
    # apply mask to volume_image
    # volume_image = np.array([make_lungmask(img) for img in volume_image ])
    
    npz_image_folder = os.path.join(train_image_npz_folder, encode)
    if not os.path.exists(npz_image_folder):
        os.mkdir(npz_image_folder) 
        
    npz_label_folder = os.path.join(train_label_npz_folder, encode)
    if not os.path.exists(npz_label_folder):
        os.mkdir(npz_label_folder) 
        
    num_slice = volume_image.shape[0]
#     num_slice = label_array.shape[0]

    for _z in range(0, num_slice):
        npz_path = os.path.join(npz_image_folder, "%03d.npz"%(_z))
        np.savez_compressed(npz_path, image=volume_image[_z])
        
    del volume_image

    for _z in range(0, num_slice):
        npz_path = os.path.join(npz_label_folder, "%03d.npz"%(_z))
        np.savez_compressed(npz_path, label=label_array[_z])
         
    del label_array

    
      
    
   
