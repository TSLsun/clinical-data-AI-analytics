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

import multiprocessing
from functools import partial

from scipy import ndimage as scndi
import random

from preprocess import load_dicom_volume, load_label
from preprocess import resample, make_lungmask, hounsfield_unit_tranformation, window_normalization

# from plotting import plot_hu_distribution, plot_ct_scan

from models import unet2D, dice_coefficient_loss, _dice_coefficient

from keras import utils as kutils
from keras.models import Model, load_model
from keras import layers as klayers
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint


# train_image_folder = "../input/train-images/image/"
# train_label_folder = "../input/train-labels/label/"

# train_list = os.listdir(train_image_folder)[:]

# bad_train = [
    # '.DS_Store',
    # '2w2suqniwvkbxnfdpi2qayhlyfouhozt',
    # '4rgpktehnvslfprnaatyi0rv75peac7o',
    # '6akhpnzfnk9cgkik8k7izk6akx0kbmxp',
    # '081j5f5m21t7nb1yd03uncoofau9cfzu',
    # 'nx5t62d6hzvm1egydllfmg8mti2omeil',
    # 'oa7nilk4nuzck787quj2yms05hz6k58s',
    # 'prmp9uolz3qje3x067gb6ze096afmj3e',
    # 'pvy9t4lp7wxyn1sn813xrqxe5e2c2l4u',
    # 'qrtx68af7g9q6hbq4g2p66itb31ym3c3',
    # 's6lyih6jfv9izt34lkah9ywva5wgeqxk',
    # 'uz15d5odukrbslyuzel6qhgo28pvf5kt',
    # 'w76zb988p62bqzxi6fprbyd7dju0l58h',
    # 'hmvsa0loxh3ek2y8rzmcyb6zrrh9mwyp', 
# ]

# Ignore this data
# if 'hmvsa0loxh3ek2y8rzmcyb6zrrh9mwyp' in train_list:
#     train_list.remove('hmvsa0loxh3ek2y8rzmcyb6zrrh9mwyp')

# for s in bad_train:
    # if s in train_list:
        # train_list.remove(s)

    
# print('Train data:', len(train_list))


#
train_image_npz_folder = '../npz-norm/train_images/'
train_label_npz_folder = '../npz-norm/train_labels/'

map_image_list = sorted(glob.glob(os.path.join(train_image_npz_folder, '*/*.npz')))
map_label_list = sorted(glob.glob(os.path.join(train_label_npz_folder, '*/*.npz')))


# split train valid data

def split_train_test(X, y, test_size=0.1, random_seed=42):
    """return train_X, train_y, test_X, test_y
    """
    random.seed(random_seed)
    test_n = int (len(X) * test_size)
    test_indices = sorted(random.sample(range(len(X)), test_n))
    
    test_X = [ X[i] for i in test_indices ]
    test_y = [ y[i] for i in test_indices ]
    train_X = [ X[i] for i in range(len(X)) if i not in test_indices ]
    train_y = [ y[i] for i in range(len(X)) if i not in test_indices ]
    
    return train_X, train_y, test_X, test_y

train_image_list, train_label_list, valid_image_list, valid_label_list = split_train_test(map_image_list, map_label_list, 0.1)


map_df = pd.DataFrame(data={'image': train_image_list, 'label': train_label_list})
map_df.head()

map_val_df = pd.DataFrame(data={'image': valid_image_list, 'label': valid_label_list})
map_val_df.head()


# ## Data generator

class LungSliceModelGenerator(kutils.Sequence):
    'Generates data for Keras'
    def __init__(self, mapping_df, batch_size, shuffle=True):
        'Initialization'
        self.mapping_df = mapping_df
        self.data_num   = mapping_df.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.data_num / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_mapping_df =             self.mapping_df.iloc[index*self.batch_size: (index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(batch_mapping_df)
        return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.mapping_df = self.mapping_df.sample(frac=1).reset_index(drop=True)
            
    def __data_generation(self, batch_mapping_df):
        'Generates data containing batch_size samples' 
        # Initialization
        X = np.zeros((self.batch_size, 512, 512, 1))
        y = np.zeros((self.batch_size, 512, 512, 1))

        # Generate data
        cnt = 0
        for i, row in batch_mapping_df.iterrows():
            X[cnt, :, :, 0] = np.load(row['image'])['image']
            y[cnt, :, :, 0] = np.load(row['label'])['label']
            
            cnt += 1
        return X, y

# model UNet 2D 
model = unet2D(depth=3)
model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=[_dice_coefficient(0.5)])
model.summary()


if not os.path.exists('../model'):
    os.mkdir('../model')
    
model_folder = os.path.join('../model', 'norm')
if not os.path.exists(model_folder):
    os.mkdir(model_folder)


callbacks = []
callbacks.append(ModelCheckpoint(os.path.join(model_folder, 'model-{epoch:03d}-{val_hard_dice_coefficient:.2f}.h5'), 
                                 save_best_only=False, 
                                 # monitor="val_hard_dice_coefficient",
                                 # mode='max',
                                 period=5))
# 
batch_size = 8
slice_generator = LungSliceModelGenerator(map_df, batch_size=batch_size)
slice_valid_generator = LungSliceModelGenerator(map_val_df, batch_size=batch_size)


# train 
history = model.fit_generator(slice_generator,
                              validation_data=slice_valid_generator,
                              epochs=25,
                              verbose=1,
                              callbacks=callbacks)

print(history.history.keys())

fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('loss_curve.png')

fig = plt.figure()
plt.plot(history.history['hard_dice_coefficient'])
plt.plot(history.history['val_hard_dice_coefficient'])
plt.title('model dice_coeffiecient')
plt.ylabel('dice_coefficient')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('dice_curve.png')


# ## Test

def retrieve_pred_str(src_dir, model, threshold=0.4):
    encode_name = src_dir.split('/')[-1]
    
    _, test_volume, info_dict = load_dicom_volume(src_dir, suffix='*.dcm')
    test_volume = hounsfield_unit_tranformation(test_volume,
                                         slope=info_dict['RescaleSlope'],
                                         intercept=info_dict['RescaleIntercept'])

    test_volume = window_normalization(test_volume)
    
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


# In[ ]:


test_image_folder = "../input/test-images/image/"


sample_submission = np.genfromtxt('../input/sample_submission.csv', 
                                  delimiter=',', 
                                  dtype='str',
                                  skip_header = 1)

test_encode_list = sample_submission[:, 0]


# In[ ]:


pretrain_model = model

pred_pair_list = []

for encode_name in tqdm.tqdm(test_encode_list, total=len(test_encode_list)):
    (encode, label_str) = retrieve_pred_str(os.path.join(test_image_folder, encode_name), 
                                            pretrain_model, 
                                            threshold=0.2)
    pred_pair_list.append((encode, label_str))


# In[ ]:


solution_path = './pred_norm_e25.csv'
with open(solution_path, 'w') as f:
    f.write('encode,pixel_value\n')
    for _pair in pred_pair_list:
        encode = _pair[0]
        label_str = _pair[1]
        f.write(encode + ',' + label_str + '\n')

