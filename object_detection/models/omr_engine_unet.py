import tensorflow as tf
import numpy as np
import os
import keras

import imageio
from PIL import Image

from tf.keras import layers
from tf.keras.models import Sequential
from tf.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate

class OMREngineUNet(tf.keras.Model):
    def __init__(self, n_blocks, p_size=None, 
                 dropout_prob=0.3, n_layers=2, n_filters=32, 
                 kernel_size=3, act_func='relu', kernel_init='HeNormal', 
                 pad='same', stride=2, n_classes=10):
        
        super(OMREngineUNet,self).__init__()

        #define encoder blocks
        self.skip_connections = []
        self.encoder_blocks = []
        
        for i in range(1,n_blocks+1):
            temp = []

            #add conv layers to blocks
            for _ in range(n_layers):
                temp.append(Conv2D(n_filters*(2**i),
                                   kernel_size,
                                   activation=act_func,
                                   padding=pad,
                                   kernel_initializer=kernel_init))

            temp.append(BatchNormalization())

            if dropout_prob > 0:
                temp.append(Dropout(dropout_prob))

            #save skip connection without max pooling to prevent loss
            self.skip_connections.append(keras.Sequential(temp))

            if p_size and i!=n_blocks:
                temp.append(MaxPooling2D(pool_size=p_size))
            
            self.encoder_blocks.append(keras.Sequential(temp))

        #define decoder blocks
        self.decoder_blocks = []
        skip_connection = 0

        for j in range(n_blocks-1,-1,-1):
            temp = []

            #merge transposer with corresponding encoder block
            temp.append(concatenate([Conv2DTranspose(n_filters*(2**j), (kernel_size, kernel_size), strides=(stride,stride), padding=pad),
                                     self.skip_connections[skip_connection]],
                                     axis=3))

            #add conv layers to blocks
            for _ in range(n_layers):
                temp.append(Conv2D(n_filters*(2**j), 
                                   kernel_size, 
                                   activation=act_func,
                                   padding=pad,
                                   kernel_initializer=kernel_init))

            self.decoder_blocks.append(keras.Sequential(temp))
        
        #add final layers for the model
        self.pen_conv = Conv2D(n_filters,
                    kernel_size,
                    activation=act_func,
                    padding='same',
                    kernel_initializer='he_normal')

        self.final_conv = Conv2D(n_classes, 1, padding='same')


    def call(self,input):
        #apply encoder and decoder blocks
        for encoder in self.encoder_blocks:
            input = encoder(input)
        
        for decoder in self.decoder_blocks:
            input = decoder(input)

        #apply final layers
        input = self.pen_conv(input)

        return self.final_conv(input)

    def pre_process(self, img):
        # Pull the relevant dimensions for image and mask
        m = len(img)                     # number of images
        i_h,i_w,i_c = target_shape_img   # pull height, width, and channels of image
        m_h,m_w,m_c = target_shape_mask  # pull height, width, and channels of mask
        
        # Define X and Y as number of images along with shape of one image
        X = np.zeros((m,i_h,i_w,i_c), dtype=np.float32)
        y = np.zeros((m,m_h,m_w,m_c), dtype=np.int32)
    
        #resize images and masks
        for file in img:
            #convert image into array of desired shape (3 channels)
            index = img.index(file)
            path = os.path.join(path1, file)
            single_img = Image.open(path).convert('RGB')
            single_img = single_img.resize((i_h,i_w))
            single_img = np.reshape(single_img,(i_h,i_w,i_c)) 
            single_img = single_img/256.
            X[index] = single_img
                
            single_mask_ind = mask[index]
            path = os.path.join(path2, single_mask_ind)
            single_mask = Image.open(path)
            single_mask = single_mask.resize((m_h, m_w))
            single_mask = np.reshape(single_mask,(m_h,m_w,m_c)) 
            single_mask = single_mask - 1 
            y[index] = single_mask
        return X,y
    
if __name__ == "__main__":    # Complete the model with 1 3x3 convolution layer (Same as the prev Conv Layers)
    # Followed by a 1x1 Conv layer to get the image to the desired size. 
    # Observe the number of channels will be equal to number of output classes
    conv9 = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(ublock9)

    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)
    pass