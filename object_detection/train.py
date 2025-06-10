import tensorflow as tf
import keras
import numpy as np

import os
import datetime
import time

from model import OMREnginePatchGan, OMREngineUNet

log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.nmow().strftime("%Y%m%d-%H%M%S")
)

discriminator, generator = OMREnginePatchGan(3), OMREngineUNet(3)


@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator.loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator.loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)
        generator.optimiser.apply_gradients(zip(generator_gradients,
                                        generator.trainable_variables))
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     discriminator.trainable_variables)

        discriminator.optimiser.apply_gradients(zip(discriminator_gradients,
                                                    discriminator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
            tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

def fit(train_ds, test_ds, steps):
    example_input, example_target = next(iter(test_ds.take(1)))
    
    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if (step) % 1000 == 0:
            display.clear_output(wait=True)

            if step!=0:
                print(f"Time taken for 1000 steps:{time.time()-start:.2f}sec \n")