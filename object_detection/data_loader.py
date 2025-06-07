import tensorflow as tf

class DataLoader:
    def __innit__(self, path, batch_size, buffer_size, height=256, width=256):
        self.path = path
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.height = height
        self.width=256
    
    def load_image_train(self, src_path, tar_path):
        src_image, tar_image = self.load_data(src_path, tar_path)
        src_image, tar_image = self.random_jitter(src_image,tar_image)
        src_image, tar_image = self.normalise(src_image, tar_image)
        return src_image, tar_image
    
    def load_iamge_test(self, src_path, tar_path):
        src_image, tar_image = self.load_data(src_path,tar_path)
        src_image, tar_image = self.normalise(src_image, tar_image)
        return src_image, tar_image        

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
    
    def resize(src_image, tar_image, height, width):
        tf.image.resize(src_image, [height, width],
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        tf.image.resize(tar_image, [height, width],
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        return src_image, tar_image

    def random_crop(src_image, tar_image):
        stacked_image = tf.stack([src_image, tar_image], axis=0)
        cropped_image = tf.image.random_crop(
            stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3]
        )
        return cropped_image[0], cropped_image[1]
    
    def normalise(src_image,tar_image):
        return (src_image/127.5)-1, (tar_image/127.5)-1
    
    @tf.function()
    def random_jitter(self,src_image, tar_image):
        #resize image
        src_image, tar_image = self.resize(src_image, tar_image, 286, 286)

        #crop image
        src_image, tar_image = self.random_crop(src_image, tar_image)

        if tf.random.uniform(()) > 0.5:
            src_image = tf.iamge.flip_left_right(src_image)
            tar_image = tf.image.flip_left_right(tar_image)
        
        return src_image, tar_image

    