import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

#subclass for music symbol detection using tf
class MusicSymbolDetectorCNN(tf.keras.Model):
    def __init__(self, num_layers, pool_size, kernel_size=3, act_func='relu',num_classes=20):
        super(MusicSymbolDetectorCNN,self).__init__()

        self.conv_blocks = []

        for j in range(1,num_layers+1):
            #initialise conv layer
            self.conv_blocks.append( tf.keras.layers.conv2D(16*(j+1),kernel_size, activation=act_func))

            #add max pooling layer every other layer
            if j % 2 == 0:
                self.conv_blocks.append(tf.keras.MaxPooling2D(pool_size))
        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes)

    #define model forward pass
    def call(self,input):
        #apply layers to input
        for layer in self.conv_bkocks:
            input = layer(input)

        #apply pooling
        input = self.global_pool(input)

        return self.classifier(input)
