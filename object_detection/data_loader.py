import tensorflow as tf
import os

class DataLoader:
    def __init__(self, batch_size, buffer_size, height=256, width=256):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.height = height
        self.width=width

    def load_imgs(self, src_img, tar_img):

        src_img,tar_img = tf.io.read_file(src_img), tf.io.read_file(tar_img)# read img as byte string
        src_img, tar_img  = tf.io.decode_jpeg(src_img), tf.io.decode_jpeg(tar_img) # convert byte string to tensor  
        src_img, tar_img = tf.cast(src_img, tf.float32), tf.cast(tar_img, tf.float32) # convert to float32 tensor
        
        return src_img, tar_img
    
    def normalise(self, src, tar):
        return (src/127.5)-1, (tar/127.5)-1
    
    def resize(self, src_image, tar_image, height=286, width=286):
        tf.image.resize(src_image, [height, width],
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        tf.image.resize(tar_image, [height, width],
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return src_image, tar_image
    
    @tf.function()
    def random_jitter(self, src, tar):

        #resize
        src, tar = self.resize(src, tar)

        #crop images
        stacked = tf.stack([src, tar],axis=0)
        cropped = tf.image.random_crop(stacked,size=[2, self.height, self.width, 3])
        src, tar = cropped[0], cropped[1]

        #mirror images
        if tf.random.uniform(()) > 0.5:
            src = tf.image.flip_left_right(src)
            tar = tf.image.flip_left_right(tar)
        
        return src, tar

    def load_and_preprocess(self, src, tar, augment=True):
        
        #load and augment
        src_img, tar_img = self.load_imgs(src,tar)
        if augment:
            src_img, tar_img = self.random_jitter(src_img, tar_img)

        src_img, tar_img = self.normalise(src_img, tar_img)

        return src_img, tar_img

    def get_dataset(self, src_path, tar_path, split='train'):
        src_files = sorted([os.path.join(src_path,f) for f in os.listdir(src_path)])
        tar_files = sorted([os.path.join(tar_path,f) for f in os.listdir(tar_path)])

        dataset = tf.data.Dataset.from_tensor_slices((src_files,tar_files))
        if split=="train":
            dataset = dataset.map(
                lambda src, tar: self.load_and_preprocess(src, tar, augment=True),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            dataset = dataset.map(
                lambda src, tar: self.load_and_preprocess(src,tar, augment=False),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

if __name__ == "__main__":
    loader = DataLoader(32,10)
    src_path = "data/input"
    tar_path = "data/target"
    data = loader.get_dataset(src_path, tar_path)
    print(data)