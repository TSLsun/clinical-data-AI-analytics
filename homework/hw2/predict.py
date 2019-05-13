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
import multiprocessing
from functools import partial

import pydicom as dicom
import nibabel as nib
from scipy import ndimage as scndi
from keras import utils as kutils

from preprocess import load_dicom_volume, load_label
from preprocess import hounsfield_unit_tranformation, window_normalization, resample, make_lungmask 
# from plotting import plot_hu_distribution, plot_ct_scan
from models import unet2D, dice_coefficient_loss, _dice_coefficient


def retrieve_pred_str(src_dir, model, threshold=0.4):
    encode_name = src_dir.split('/')[-1]
    
    _, test_volume, info_dict = load_dicom_volume(src_dir, suffix='*.dcm')
    test_volume = hounsfield_unit_tranformation(test_volume,
                                         slope=info_dict['RescaleSlope'],
                                         intercept=info_dict['RescaleIntercept'])
    test_volume = window_normalization(test_volume, lower_th=-1000., upper_th=400)
    # test_volume, _ = resample(test_volume, original_spacing=info_dict['Spacing'])
    # test_volume = np.array([make_lungmask(img) for img in test_volume ])
    # test_volume = np.array([resize(img, (512,512), anti_aliasing=True) for img in test_volume]) 
    
    pred_label = model.predict(np.expand_dims(test_volume, axis=-1))
    pred_label = np.transpose(pred_label[:, :, :, 0], axes=(2, 1, 0))
    
    pred_label = (pred_label > threshold).astype(np.int)

    label_flatten = pred_label.flatten()

    label_flatten_idx = np.where(label_flatten == 1)[0]

    label_str = ''
    
    if label_flatten_idx.size > 0:
        prev_idx = label_flatten_idx[0]
        idx_start = label_flatten_idx[0]
        cnt = 1
        for _idx in label_flatten_idx[1:]:
            if _idx == prev_idx+1:
                cnt += 1
            else:
                label_str += str(idx_start) + ' ' + str(cnt) + ' '

                cnt = 1
                idx_start = _idx
            prev_idx = _idx

        label_str = label_str.rstrip(' ')
    return (encode_name, label_str)


def main(args):

    # test_image_folder = "../input/test-images/image/" 
    test_image_folder = args.test_image_folder 
    sample_submission = np.genfromtxt('sample_submission.csv', 
                                      delimiter=',', 
                                      dtype='str',
                                      skip_header = 1)



    test_encode_list = sample_submission[:, 0]


    # from keras.models import load_model

    # model_weight = 'model/sample-code/model-030-0.00.h5'
    model_weight = args.model_dir
    pretrain_model = unet2D(pretrained_weights=model_weight, depth=args.unet_depth)


    pred_pair_list = []

    for encode_name in tqdm.tqdm(test_encode_list, total=len(test_encode_list)):
        (encode, label_str) = retrieve_pred_str(os.path.join(test_image_folder, encode_name), 
                                                pretrain_model, 
                                                threshold=0.2)
        pred_pair_list.append((encode, label_str))


    # solution_path = './sample-code_pred_e30.csv'
    solution_path = args.predict_csv_dir
    with open(solution_path, 'w') as f:
        f.write('encode,pixel_value\n')
        for _pair in pred_pair_list:
            encode = _pair[0]
            label_str = _pair[1]
            f.write(encode + ',' + label_str + '\n')

    logging.info("Done prediction... save to " + args.predict_csv_dir)

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script to predict.")
    parser.add_argument('test_image_folder', type=str,
                        help='Directory to the test-images folder.')
    parser.add_argument('predict_csv_dir', type=str)
    parser.add_argument('--model_dir', type=str, default='model.h5',
                        help='Directory to the model checkpoint.')
    parser.add_argument('--unet_depth', type=int, default=3)
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
