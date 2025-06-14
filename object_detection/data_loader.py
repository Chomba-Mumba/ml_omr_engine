import tensorflow as tf
import os

class DataLoader:
    def __init__(self, batch_size, buffer_size, height=1024, width=1024):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.height = height
        self.width=width

    def load_imgs(self, src_img, tar_img):
        
        src_img,tar_img = tf.io.read_file(src_img), tf.io.read_file(tar_img)# read img as byte string
        src_img, tar_img  = tf.io.decode_png(src_img , channels=3), tf.io.decode_png(tar_img, channels=3) # convert byte string to tensor 
        src_img, tar_img = tf.cast(src_img, tf.float32), tf.cast(tar_img, tf.float32) # convert to float32 tensor

        #resize
        src_img, tar_img = self.resize(src_img, tar_img, self.height, self.width)
        print(src_img.shape)
        
        return src_img, tar_img
    
    def normalise(self, src, tar):
        return (src/127.5)-1, (tar/127.5)-1
    
    def resize(self, src_image, tar_image, height=286, width=286):
        src_image = tf.image.resize(src_image, [height, width],
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        tar_image = tf.image.resize(tar_image, [height, width],
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return src_image, tar_image
    
    @tf.function()
    def random_jitter(self, src, tar):

        if tf.random.uniform(()) > 0.5:
            #crop images
            stacked = tf.stack([src, tar],axis=0)
            cropped = tf.image.random_crop(stacked,size=[2, self.height, self.width, 3])
            src, tar = cropped[0], cropped[1]

            #mirror images
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
    
    def split(self, dataset, train_size=0.7, test_size=0.15, val_size=0.15):
        dataset.shuffle()

        dataset_size = dataset.cardinality().numpy()

        train_size = train_size * dataset_size 
        test_size = test_size * dataset_size 
        val_size = val_size * dataset_size 

        train_ds = dataset.take(train_size * dataset_size )
        test_ds = dataset.skip(train_size)
        val_ds = test_ds.skip(val_size)
        test_ds= test_ds.take(test_size)

        return train_ds, test_ds, val_ds

    def get_dataset(self, src_path, tar_path, split='train'):
        src_files = sorted([os.path.join(src_path,f) for f in os.listdir(src_path)])
        tar_files = sorted([os.path.join(tar_path,f) for f in os.listdir(tar_path)])

        dataset = tf.data.Dataset.from_tensor_slices((src_files,tar_files))
        dataset = dataset.map(
            lambda src, tar: self.load_and_preprocess(src, tar, augment=True),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

if __name__ == "__main__":
    loader = DataLoader(3,10)

    src_path = "data/input"
    tar_path = "data/target"

    data = loader.get_dataset(src_path, tar_path)
    print(f"data specs:{data.element_spec}")

    # preview tensors
    print(f"data as iter: {list(data.as_numpy_iterator())}")