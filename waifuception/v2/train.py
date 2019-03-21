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

import model as waifuception
import input_proc

def main():
    base_lr = 0.045
    batch_size = 32
    weights_dir = '/mnt/data/waifuception-v2-checkpoints/'

    try:
        if osp.isdir('/mnt/data/tensorboard-logs'):
            shutil.rmtree('/mnt/data/tensorboard-logs')

        os.mkdir('/mnt/data/tensorboard-logs')
    except OSError:
        print("Warning: could not clear tensorboard data")

    try:
        if not osp.isdir(weights_dir):
            os.mkdir(weights_dir)
    except OSError:
        print("Warning: could not make checkpoints dir")
        
    train_dataset, n_batches_train, eval_dataset, n_batches_eval = input_proc.setup_input_pipeline(
        sys.argv[1],
        10000,
        batch_size
    )

    print("Building model...")
    model, resume_from_epoch = waifuception.build_model(base_lr, weights_dir, base_lr)

    print("Starting training.")
    model.fit(
        train_dataset.make_one_shot_iterator(),
        steps_per_epoch  = n_batches_train,
        validation_data  = eval_dataset.make_one_shot_iterator(),
        validation_steps = n_batches_eval,
        epochs           = 200,
        verbose          = 1,
        initial_epoch    = resume_from_epoch,
        callbacks=[
            callbacks.ModelCheckpoint(weights_dir+'weights.{epoch:03d}.{val_loss:.04f}.hdf5'),
            callbacks.LearningRateScheduler(lambda epoch, cur_lr: base_lr * np.power(0.87, epoch)),
            #callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001),
            callbacks.TensorBoard('/mnt/data/tensorboard-logs', update_freq=700),
        ]
    )

if __name__ == '__main__':
    main()
