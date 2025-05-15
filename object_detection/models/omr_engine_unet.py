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
    def __init__(self, n_blocks, p_size=None, dropout_prob=0.3, n_layers=2, n_filters=32, kernel_size=3, act_func='relu', kernel_init='HeNormal',pad='same'):
        super(OMREngineUNet,self).__init__()
        #define encoder blocks
        self.skip_connections = []
        self.encoder_blocks = []
        
        for _ in range(n_blocks):
            temp = []
            #add conv layers to blocks
            for j in range(1,n_layers+1):
                #init conv layer
                temp.append(Conv2D(16*(j+1),kernel_size, activation=act_func))

            #add batch normalisation to normalise the otput of the last layer
            temp.append(BatchNormalization())

            #prevent overfiting with dropout
            if dropout_prob > 0:
                temp.append(Dropout(dropout_prob))

            #save skip connection without max pooling to prevent loss
            self.skip_connections.append(keras.Sequential(temp))

            #max pooling for dimensionality reduction
            if p_size:
                temp.append(MaxPooling2D(pool_size=p_size))
            
            #add block to encoder block
            self.encoder_blocks.append(keras.Sequential(temp))
            
        #define decoder blocks
        self.decoder_blocks = []
        for i in range(n_blocks):
            temp = []
            #merge transposer with corresponding encoder block
            temp.append(concatenate([Conv2DTranspose(16*(j+1),kernel_size, activation=act_func),self.skip_connections[i]]))

            #add conv layers to blocks
            for j in range(1,n_layers+1):
                #init conv layer
                temp.append(Conv2D(16*(j+1),kernel_size, activation=act_func))

            #add block to encoder block
                self.encoder_blocks.append(keras.Sequential(temp))



    def call(self,input):
        #apply layers to input
        for layer in self.encoder_blocks:
            input = layer(input)

        #apply pooling
        input = self.global_pool(input)

        return self.classifier(input)

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
    
if __name__ == "__main__":
    pass