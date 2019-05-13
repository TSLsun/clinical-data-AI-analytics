import numpy as np
from keras.models import Model, load_model
from keras import layers as klayers
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint

# ## Loss function

def _dice_coefficient(threshold = 0.3):
    def hard_dice_coefficient(y_true, y_pred, smooth=1.0):
        y_true_f = K.flatten(K.cast(y_true > threshold, dtype=float))
        y_pred_f = K.flatten(K.cast(y_pred > threshold, dtype=float))
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return hard_dice_coefficient

def dice_coefficient_loss(y_true, y_pred):
    return 1 - _dice_coefficient()(y_true, y_pred)


# ## Model architecture

def unet2D(pretrained_weights=None, input_size=[512, 512, 1], depth=3, init_filter=8, 
         filter_size=3, padding='same', pool_size=[2, 2], strides=[2, 2]):
    
    inputs = klayers.Input(input_size)
    
    current_layer = inputs
    encoding_layers = []
    
    # Encoder path
    for d in range(depth + 1):
        num_filters = init_filter * 2 ** d
        
        conv = klayers.Conv2D(num_filters, filter_size, padding=padding, kernel_initializer='he_normal')(current_layer)
        conv = klayers.BatchNormalization()(conv)
        conv = klayers.Activation('relu')(conv)
        conv = klayers.Conv2D(num_filters * 2, filter_size, padding=padding, kernel_initializer='he_normal')(conv)
        conv = klayers.BatchNormalization()(conv)
        conv = klayers.Activation('relu')(conv)
        encoding_layers.append(conv)
    
        pool = klayers.MaxPooling2D(pool_size=pool_size)(conv)
        
        if d == depth:
            # Bridge
            current_layer = conv
        else:
            current_layer = pool

        
    # Decoder path
    for d in range(depth, 0, -1):
        num_filters = init_filter * 2 ** d
        up = klayers.Deconvolution2D(num_filters * 2, pool_size, strides=strides)(current_layer)

        crop_layer = encoding_layers[d - 1]
        # Calculate two layers shape
        up_shape = np.array(up._keras_shape[1:-1])
        conv_shape = np.array(crop_layer._keras_shape[1:-1])

        # Calculate crop size of left and right
        crop_left = (conv_shape - up_shape) // 2

        crop_right = (conv_shape - up_shape) // 2 + (conv_shape - up_shape) % 2
        crop_sizes = tuple(zip(crop_left, crop_right))

        crop = klayers.Cropping2D(cropping=crop_sizes)(crop_layer)

        # Concatenate
        up = klayers.Concatenate(axis=-1)([crop, up])
        conv = klayers.Conv2D(num_filters, filter_size, padding=padding, kernel_initializer='he_normal')(up)
        conv = klayers.BatchNormalization()(conv)
        conv = klayers.Activation('relu')(conv)
        conv = klayers.Conv2D(num_filters, filter_size, padding=padding, kernel_initializer='he_normal')(conv)
        conv = klayers.BatchNormalization()(conv)
        conv = klayers.Activation('relu')(conv)
        
        current_layer = conv
    
    
    outputs = klayers.Conv2D(1, 1, padding=padding, kernel_initializer='he_normal')(current_layer)
    outputs = klayers.Activation('sigmoid')(outputs)
    model = Model(inputs=inputs, outputs=outputs)

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

