import tensorflow as tf
import numpy as np
import os
import keras

from data_loader import DataLoader
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate

class OMREnginePatchGan(tf.keras.Model):
    def __init__(self, n_blocks, n_filters=32, 
                 kernel_size=3, pad='same', 
                 stride=(2,2), n_classes=10):
        
        super(OMREnginePatchGan, self).__init__()
        kernel_init = tf.random_normal_initializer(0.,0.02)

        input = tf.keras.layers.Input(shape=[256,256,3], name="input_image")
        target = tf.keras.layers.Input(shape=[256,256,3], name="target_image")

        x = tf.keras.layers.concatenate([input,target])

        #define encoder blocks
        self.encoder_blocks = []

        for j in range(1,n_blocks+1):
            block = []

            if j == 1:
                block.append(Conv2D(n_filters*(2**j), 
                                kernel_size, strides=stride,
                                padding=pad, kernel_initializer=kernel_init, use_bias=False))
                block.append(x)
            else:
                block.append(Conv2D(n_filters*(2**j), 
                                kernel_size, strides=stride,
                                padding=pad, kernel_initializer=kernel_init, use_bias=False))
                
                block.append(tf.keras.layers.BatchNormalization())

            block.append(tf.keras.layers.LeakyReLU())
            self.encoder_blocks.append(keras.Sequential(block))
        

        #add final layers for the model
        self.pen_conv = Conv2D(n_filters*(2**n_blocks+2), kernel_size, strides=1,
                    kernel_initializer=kernel_init,
                    padding='same',
                    use_bias=False)
        
        self.final_conv = Conv2D(n_classes, 1, padding='same', strides=1)

        #initialise loss
        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


    def call(self,input):
        
        for encoder in self.encoder_blocks:
            input = encoder(input)

        input = tf.keras.layers.ZeroPadding2D()(input)
        input = self.pen_conv(input)

        input = tf.keras.layers.BatchNormalization()(input)
        input = tf.keras.layers.LeakyReLU()(input)
        input = tf.keras.layers.ZeroPadding2D()(input)

        return self.final_conv(input)
    
    def loss(self, real_output, gen_output):
        real_loss = self.loss_object(tf.ones_like(real_output), real_output)

        generated_loss = self.loss_object(tf.zeros_like(gen_output), gen_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss
    def save_checkpoint(self,generator_optimiser, discriminator_optimiser, generator, time):
        #checkpoint
        checkpoint_dir="./checkpoints"
        checkpoint_prefix = os.path.join(checkpoint_dir,"chkpt")

        self.checkpoint=tf.train.Checkpoint(
            generator_optimizer = generator_optimiser,
            discriminator_otpmiser = discriminator_optimiser,
            generator = generator,
            discriminator=self)

        self.checkpoint.save(file_prefix=checkpoint_prefix)
        print(f"Checkpoint saved at: {time}")
    

class OMREngineUNet(tf.keras.Model):
    def __init__(self, n_blocks, dropout_prob=0.3, n_filters=32, 
                 kernel_size=4, act_func='relu',
                 pad='same', stride=2, n_classes=10):
        
        super(OMREngineUNet, self).__init__()

        #define encoder blocks
        self.encoder_blocks = []

        initialiser = tf.random_normal_initializer(0., 0.02)
        
        for i in range(1,n_blocks+1):
            block = []

            block.append(Conv2D(n_filters*(2**i),
                                kernel_size, strides=stride, padding=pad,
                                kernel_initializer=initialiser,
                                use_bias=False))
            #apply batchnorm after 1st block
            if i > 1:
                block.append(BatchNormalization())

            block.append(tf.keras.layers.LeakyReLU())
            
            self.encoder_blocks.append(keras.Sequential(block))

        #define decoder blocks
        self.decoder_blocks = []

        for j in range(n_blocks-1, 0, -1):
            block = []
            #merge transposer with corresponding encoder block
            block.append(Conv2DTranspose(n_filters*(2**j), kernel_size, strides=stride, 
                                         padding=pad,
                                         kernel_initializer=initialiser,
                                         use_bias=False))

            block.append(tf.keras.layers.BatchNormalization())

            #apply dropout to first 3 layers
            if j >= n_blocks-3:
                block.append(tf.keras.layers.Dropout(dropout_prob))
            
            block.append(tf.keras.layers.ReLU())

            self.decoder_blocks.append(keras.Sequential(block))

        #add final layers for the model
        self.pen_conv = Conv2D(n_filters,
                    kernel_size,
                    activation=act_func,
                    padding='same',
                    kernel_initializer='he_normal')

        self.final_conv = Conv2D(n_classes, 1, padding='same')

        #initilise loss
        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


    def call(self,input):
        #apply encoder and decoder blocks
        skip_conns = []
        for encoder in self.encoder_blocks:
            input = encoder(input)
            print(f"input shape:{input.shape}")
            skip_conns.append(input)

        skip_conns= reversed(skip_conns[:-1])
        
        for decoder, skip in zip(self.decoder_blocks, skip_conns):
            input = decoder(input)
            input = tf.keras.layers.Concatenate()([input,skip])
            
        #apply final layers
        input = self.pen_conv(input)

        return self.final_conv(input)

    def loss(self, disc_gen_out, gen_out, target, LAMBDA=100):
        gan_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
            tf.ones_like(disc_gen_out), disc_gen_out
        )
        #MAE
        l1_loss = tf.reduce_mean(tf.abs(target-gen_out))

        total_gen_loss = gan_loss + (LAMBDA * l1_loss)

        return total_gen_loss, gan_loss, l1_loss
    
    def generate_images(self, input, target):
        prediction = self(input, training=True)

        display_list = [input[0], target[0], prediction[0]]

        for i in range(3):
            print(display_list[i] * 0.5 + 0.5)
    
if __name__ == "__main__":
    unet = OMREngineUNet(3)
    unet.summary()

    print("=============")
    test_data = tf.random.uniform((1, 128, 128, 1))
    out = unet(test_data)
    print(f"output: {out}")
    print("=============")

    patchGan = OMREnginePatchGan(3)
    patchGan.summary()
    print(patchGan)