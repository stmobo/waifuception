import math
import os
import os.path as osp
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from keras import callbacks
import keras.backend as K
import traceback

import model as waifuception
import input_proc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)

def dataset_to_iterator(ds):
    iterator = ds.make_one_shot_iterator()
    get_next = iterator.get_next()

    n_failures = 0
    while True:
        try:
            *inputs, labels = K.get_session().run(get_next)
            yield inputs, labels
            n_failures = 0
        except:
            t, v, tb = sys.exc_info()
            print(str(v))

            n_failures += 1

            if n_failures > 3:
                print("Failed 3 times to get data, crashing here...")
                raise
def main():
    base_lr = 0.045
    batch_size = 64
    weights_dir = '/mnt/data/waifuception-v2-checkpoints/2/'

    dataset_length = 331978

    try:
        if osp.isdir('/mnt/data/tensorboard-logs/2'):
            shutil.rmtree('/mnt/data/tensorboard-logs/2')

        os.mkdir('/mnt/data/tensorboard-logs/2')
    except OSError:
        print("Warning: could not clear tensorboard data")

    try:
        if not osp.isdir(weights_dir):
            os.makedirs(weights_dir)
    except OSError:
        print("Warning: could not make checkpoints dir")

    with tf.device('/cpu:0'):
        print("Building input pipeline...")
        train_dataset, n_batches_train, eval_dataset, n_batches_eval = input_proc.setup_input_pipeline(
            sys.argv[1],
            dataset_length,
            batch_size
        )

        iter_train = dataset_to_iterator(train_dataset)
        iter_eval = dataset_to_iterator(eval_dataset)

        print("Building model...")
        model, resume_from_epoch = waifuception.build_model(weights_dir)

    parallel_model = waifuception.ModelMGPU(model, 4)

    next(iter_train)
    next(iter_eval)
    waifuception.compile_model(parallel_model, base_lr)

    print("Starting training.")
    parallel_model.fit_generator(
        iter_train,
        steps_per_epoch  = n_batches_train,
        validation_data  = iter_eval,
        validation_steps = n_batches_eval,
        epochs           = 200,
        verbose          = 1,
        initial_epoch    = resume_from_epoch,
        callbacks=[
            callbacks.ModelCheckpoint(osp.join(weights_dir, 'weights.{epoch:03d}.{val_loss:.04f}.hdf5')),
            callbacks.LearningRateScheduler(lambda epoch, cur_lr: base_lr * np.power(0.87, (epoch//2))),
            #callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001),
            callbacks.TensorBoard('/mnt/data/tensorboard-logs/2', update_freq=700),
        ]
    )

if __name__ == '__main__':
    main()
