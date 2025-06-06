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
    def __init__(self, n_blocks, p_size=(2,2), 
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
            
            #add pooling to all encoder blocsk except last
            if i!=n_blocks:
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

            skip_connection += 1

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

        #initilise loss
        self.loss_obj = tf.keras.losses.BinaryCrossEntropy(from_logits=True)


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

    def pre_process(self, target_shape_img, target_shape_mask, image_path, mask_path):

        imgs,masks = self.load_data(image_path, mask_path)

        # Pull the relevant dimensions for image and mask
        n_imgs = len(imgs)
        i_h,i_w,i_c = target_shape_img   
        m_h,m_w,m_c = target_shape_mask
        
        # define X and y as number of images along with shape of one image
        X = np.zeros((n_imgs,i_h,i_w,i_c), dtype=np.float32)
        y = np.zeros((n_imgs,m_h,m_w,m_c), dtype=np.int32)
    
        #resize images and masks
        for file in imgs:
            #convert image into array of desired shape (3 channels)
            index = imgs.index(file)
            path = os.path.join(image_path, file)
            single_img = Image.open(path).convert('RGB')
            single_img = single_img.resize((i_h,i_w))
            single_img = np.reshape(single_img,(i_h,i_w,i_c)) 
            single_img = single_img/256.
            X[index] = single_img
                
            single_mask_ind = masks[index]
            path = os.path.join(mask_path, single_mask_ind)
            single_mask = Image.open(path)
            single_mask = single_mask.resize((m_h, m_w))
            single_mask = np.reshape(single_mask,(m_h,m_w,m_c)) 
            single_mask = single_mask - 1 
            y[index] = single_mask
        return X,y
    
    def generator_loss(disc_generated_output, gen_output, target):
        gan_loss = self.loss_obj(tf.ones_like(disc_generated_output), disc_generated_output)

        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (LAMBDA * l1_loss)

        return total_gen_loss, gan_loss, l1_loss
    
if __name__ == "__main__":    
    pass