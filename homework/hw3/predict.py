#!/usr/bin/env python
# coding: utf-8

# ## Load modules

import argparse
import logging
import pdb
import pickle
import sys
import traceback

import os
import glob
import tqdm
import numpy as np 
import pandas as pd
# import matplotlib.pyplot as plt

import pydicom as dicom
import nibabel as nib
import skimage

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

vgg = VGG16(weights='imagenet', include_top=False)

def load_nii_gz(label_fpath, transpose=False):
    data = nib.load(label_fpath)
    pixel_array = data.get_fdata()
    if transpose:
        pixel_array = np.transpose(pixel_array, axes=(2,1,0))
    return pixel_array 

from keras.models import Model
from keras.layers import Dense, Dropout, Input, Activation
from keras.models import load_model

def dnn():
    inputs = Input(shape=(3888641,))
    x = Dense(16, kernel_initializer='normal', activation='relu')(inputs)
    outputs = Dense(1, kernel_initializer='normal')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def main(args):
    model = load_model(args.model_dir)
    
    # test_image_folder = "../input/test-images/image/" 
    test_image_folder = args.test_image_folder 

    sample_submission = np.genfromtxt('sample_submission.csv', 
                                      delimiter=',', 
                                      dtype='str',
                                      skip_header = 1)
    test_encode_list = sample_submission[:, 0]

    test_X_flair = np.zeros((len(test_encode_list), 155, 240, 240))
    # test_X_seg = np.zeros((len(test_encode_list), 155, 240, 240, 1))

    for i, encode in enumerate(test_encode_list):
        test_X_flair[i, :, :, :] = load_nii_gz(os.path.join(test_image_folder, encode, 'flair.nii.gz'), transpose=True)

    test_flair = []
    for x in test_X_flair:
        x = skimage.transform.resize(x, (155,224,224))
        x = skimage.color.gray2rgb(x) 
        test_flair.append(vgg.predict(x).reshape(155,-1))
    test_X_flair = np.array(test_flair)
    
    # test_df = pd.read_csv("../../test-age.csv")
    test_df = pd.read_csv(args.test_age_csv)
    test_df = test_df.drop(columns=["Unnamed: 0"])
    test_df.reset_index(drop=True, inplace=True)

    test_X_age = np.array(test_df['Age'])
    test_X_age = test_X_age.reshape(20,1)

    test_X_flair_age = np.concatenate((test_X_flair.reshape(20,-1), test_X_age), axis = 1)

    test_preds = model.predict(test_X_flair_age)


    # solution_path = './sample-code_pred_e30.csv'
    solution_path = args.predict_csv_dir
    with open(solution_path, 'w') as f:
        f.write('Encode,Survival\n')
        for i, encode in enumerate(test_encode_list):
            print(encode, test_preds[i][0]) 
            f.write(encode + ',' + str(test_preds[i][0]) + '\n')

    logging.info("Done prediction... save to " + args.predict_csv_dir)

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script to predict.")
    parser.add_argument('test_image_folder', type=str,
                        help='Directory to the test-images folder.')
    parser.add_argument('test_age_csv', type=str,
                        help='./Path to the test-age csv file.')
    parser.add_argument('predict_csv_dir', type=str)
    parser.add_argument('--model_dir', type=str, default='model.h5',
                        help='Directory to the model checkpoint.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
