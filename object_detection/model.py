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
        self.pen_conv = Conv2D(n_filters*(2**n_blocks+2),
                    kernel_size,
                    activation=act_func,
                    padding='same',
                    kernel_initializer='he_normal')
        
        #add leaky relu
        self.leaky_relu = tf.keras.Layers.LeakyReLU()

        #add zero padding
        self.zero_pad2 = tf.keras.Layers.ZeroPadding2D()

        self.final_conv = Conv2D(n_classes, 1, padding='same', strides=1)

        #initialise loss
        self.loss_obj = tf.keras.losses.BinaryCrossEntropy(from_logits=True)


    def call(self,input):
        
        for decoder in self.decoder_blocks:
            input = decoder(input)

        #apply final layers
        input=self.zero_pad1(input)
        input = self.pen_conv(input)

        input = self.leaky_relu(input)
        input = self.zero_pad2(input)

        return self.final_conv(input)
    
    def loss(self, real_output, gen_output):
        real_loss = self.loss_object(tf.ones_like(real_output), real_output)

        generated_loss = self.loss_object(tf.zeros_like(gen_output), gen_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss
    
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
    
    def load_data(self, src_image, tar_image):
        #read the image as byte string
        src_image = tf.io.read_file(src_image)
        src_image  = tf.io.decode_jpeg(src_image) #convert byte string to tensor

        tar_image = tf.io.read_file(tar_image)
        tar_image  = tf.io.decode_jpeg(tar_image) 

        #convert to float32 tensors
        src_image = tf.cast(src_image, tf.float32)
        tar_image = tf.cast(tar_image, tf.float32)
        
        return src_image, tar_image

    def pre_process(self, src_image, tar_image):
        pass
    
    def generator_loss(self,disc_generated_output, gen_output, target):
        gan_loss = self.loss_obj(tf.ones_like(disc_generated_output), disc_generated_output)

        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (LAMBDA * l1_loss)

        return total_gen_loss, gan_loss, l1_loss