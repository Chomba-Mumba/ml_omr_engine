import tensorflow as tf
import numpy as np
import os
import keras

from tf.keras import layers
from tf.keras.models import Sequential
from tf.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate

class OMREnginePatchGan(tf.keras.Model):
    def __init__(self, n_blocks, p_size=(2,2), 
                 dropout_prob=0.3, n_layers=2, n_filters=32, 
                 kernel_size=3, act_func='relu', 
                 pad='same', stride=2, n_classes=10):
        
        super(OMREnginePatchGan,self).__init__()

        kernel_init = tf.random_normal_initializer(0.,0.02)

        input = tf.keras.Layers.Input(shape=[256,256,3], name="input_image")
        target = tf.keras.Layers.Input(shape=[256,256,3], name="target_image")

        x = tf.keras.layers.concatenate([input,target])

        #define decoder blocks
        self.decoder_blocks = []

        for j in range(1,n_blocks+1):
            temp = []

            #add conv layers to blocks
            for _ in range(n_layers):
                temp.append(Conv2D(n_filters*(2**j), 
                                   kernel_size,
                                   stride=stride, 
                                   activation=act_func,
                                   padding=pad,
                                   kernel_initializer=kernel_init, use_bias=False))

            self.decoder_blocks.append(keras.Sequential(temp))
        
        #add zero padding
        self.zero_pad1 = tf.keras.Layers.ZeroPadding2D()

        #add final layers for the model
        self.pen_conv = Conv2D(n_filters,
                    kernel_size,
                    activation=act_func,
                    padding='same',
                    kernel_initializer='he_normal')
        
        #add leaky relu
        self.leaky_relu = tf.keras.Layers.LeakyReLU()

        #add zero padding
        self.zero_pad2 = tf.keras.Layers.ZeroPadding2D()

        self.final_conv = Conv2D(n_classes, 1, padding='same', strides=1)


    def call(self,input):
        #apply encoder and decoder blocks
        for encoder in self.encoder_blocks:
            input = encoder(input)
        
        for decoder in self.decoder_blocks:
            input = decoder(input)

        #apply final layers
        input = self.pen_conv(input)

        return self.final_conv(input)
    
    def load_data(self, image_path, mask_path):
        #read the images folder like a list
        images = os.listdir(image_path)
        masks = os.listdir(mask_path)

        orig_imgs = []
        mask_imgs = []
        for file in images:
            orig_imgs.append(file)
        for file in masks:
            mask_imgs.append(file)

        orig_imgs.sort()
        mask_imgs.sort()
    
        return orig_imgs, mask_imgs

if __name__ == "__main__":
    pass