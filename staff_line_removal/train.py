import tensorflow as tf

import numpy as np
import os
import datetime
import time

from model import OMREnginePatchGan, OMREngineUNet
from data_loader import DataLoader

log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)

loader = DataLoader(1,2)

base_dir = os.getenv("BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(base_dir, "data", "input")
tar_path = os.path.join(base_dir, "data", "target")

data = loader.get_dataset(src_path, tar_path)
train_ds, test_ds, val_ds = loader.split(data)

discriminator, generator = OMREnginePatchGan(3), OMREngineUNet(3)

@tf.function
def train_step(input_image, target, step, curr_time):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        #generate image and analyse discriminator 
        gen_output = generator(input_image, training=True)
        print("Discriminating real output...")

        disc_real_output = discriminator([input_image, target], training=True)
        print("Discriminating generated output...")
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator.generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)
        generator.optimiser.apply_gradients(zip(generator_gradients,
                                        generator.trainable_variables))
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     discriminator.trainable_variables)

        discriminator.optimiser.apply_gradients(zip(discriminator_gradients,
                                                    discriminator.trainable_variables))
        
        #update checkpoints if needed
        if generator.total_loss > gen_total_loss:
            generator.save_checkpoint(best_chkpt=True)
            print(f"New best generator weights found at {curr_time} saving model...")

        if discriminator.total_loss > disc_loss:
            discriminator.save_checkpoint(best_chkpt=True)
            print(f"New best discriminator weights found at {curr_time} saving model...")

        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
            tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

def fit(train_ds, test_ds, steps):
    start = time.time()
    

    for step, (input_image, target) in enumerate(train_ds.repeat().take(steps)):
        if (step) % 1000 == 0:
            if step!=0:
                print(f"Time taken for 1000 steps:{time.time()-start:.2f}sec \n")
            start = time.time()

            print(f"Step: {step//1000}k")

        curr_time = f"{time.time()-start:.2f}sec \n"
        train_step(input_image, target, step, curr_time)

        #training step
        if (step + 1) % 10 == 0:
            print('.',end='',flush=True)
        
        #save checkpoint every 200 steps
        if (step + 1) % 200 == 0:
            generator.save_checkpoint()
            discriminator.save_checkpoint()

            print(f"Checkpoints saved at time: {time.time()-start:.2f}sec \n")

fit(train_ds, test_ds, len(train_ds))

print("Successfully trained GAN with best training weights saved.")

for inp, tar in test_ds:
    generator.generate_images(inp, tar, "test")