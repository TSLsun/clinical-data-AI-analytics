import keras
from keras.layers import Input ,Conv3D, BatchNormalization, Activation, MaxPooling3D, Cropping3D, Concatenate

def AIMIANet(input_shape=[96, 96, 96, 1], num_classes=2):
    inputs = Input(shape=input_shape, name="input")
    k3_c16_1 = Conv3D(kernel_size=(3, 3, 3), filters=16, name='k3_c16_1')(inputs)
    bn_1 = BatchNormalization(name='bn_1')(k3_c16_1)
    k3_c32_1 = Conv3D(kernel_size=(3, 3, 3), filters=32, name='k3_c32_1')(bn_1)
    bn_2 = BatchNormalization(name='bn_2')(k3_c32_1)
    relu_1 = Activation(activation='relu', name='relu_1')(bn_2)
    k2_s2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name='k2_s2')(relu_1)
    k3_c32_2 = Conv3D(kernel_size=(3, 3, 3), filters=32, name='k3_c32_2')(k2_s2)
    bn_3 = BatchNormalization(name='bn_3')(k3_c32_2)
    k3_c64_1 = Conv3D(kernel_size=(3, 3, 3), filters=64, name='k3_c64_1')(bn_3)
    bn_4 = BatchNormalization(name='bn_4')(k3_c64_1)
    relu_2 = Activation(activation='relu', name='relu_2')(bn_4)
    k3_c64_2 = Conv3D(kernel_size=(3, 3, 3), filters=64, name='k3_c64_2')(relu_2)
    bn_5 = BatchNormalization(name='bn_5')(k3_c64_2)
    k3_c128_1 = Conv3D(kernel_size=(3, 3, 3), filters=128, name='k3_c128_1')(bn_5)
    bn_6 = BatchNormalization(name='bn_6')(k3_c128_1)
    crop27 = Cropping3D(cropping=27, name='crop27')(relu_1)
    crop2 = Cropping3D(cropping=2, name='crop2')(relu_2)
    relu_3 = Activation(activation='relu', name='relu_3')(bn_6)
    concat = Concatenate(name='concat')([crop27, crop2, relu_3])
    k3_c128_2 = Conv3D(kernel_size=(3, 3, 3), filters=128, name='k3_c128_2')(concat)
    bn_7 = BatchNormalization(name='bn_7')(k3_c128_2)
    k3_c64_3 = Conv3D(kernel_size=(3, 3, 3), filters=64, name='k3_c128_3')(bn_7)
    bn_8 = BatchNormalization(name='bn_8')(k3_c64_3)
    k3_c32_3 = Conv3D(kernel_size=(3, 3, 3), filters=32, name='k3_c32_3')(bn_8)
    bn_9 = BatchNormalization(name='bn_9')(k3_c32_3)
    relu_4 = Activation(activation='relu', name='relu_4')(bn_9)
    output = Conv3D(kernel_size=1, filters=num_classes, name='k1_c_numclasses')(relu_4)
    
    model = keras.Model(inputs=inputs, outputs=output)

    
    return model