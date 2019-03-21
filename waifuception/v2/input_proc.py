import math
import os
import os.path as osp
import shutil
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import tensorflow as tf

import classes

def _parse_proto(example_proto):
    features = {
        'filename' : tf.FixedLenFeature((), tf.string, default_value=""),
        'labels' :   tf.FixedLenFeature((107,), tf.int64, default_value=np.zeros(107)),
    }

    parsed_features = tf.parse_single_example(example_proto, features)
    raw_data = tf.io.read_file(tf.string_join(['/mnt/data/dl/danbooru2018/original/', parsed_features['filename']]))
    labels = tf.cast(parsed_features['labels'], tf.float32)

    # drop the 'female' tag (assume it's 1-p(male))
    labels = labels[1:]

    img = tf.io.decode_image(raw_data, channels=3)
    img.set_shape([None, None, 3])

    img = tf.image.resize(img, (224, 224))
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = (img - 0.5) * 2.0

    return img, labels

def setup_input_pipeline(records_file, dataset_length, batch_size, val_split=0.1):
    dataset = tf.data.TFRecordDataset(records_file)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    dataset = dataset.take(dataset_length)

    eval_len = int(dataset_length * val_split)
    train_len = dataset_length - eval_len

    eval_dataset  = dataset.take(eval_len)
    eval_dataset  = eval_dataset.apply(tf.data.experimental.shuffle_and_repeat(500))
    eval_dataset  = eval_dataset.apply(tf.data.experimental.map_and_batch(_parse_proto, batch_size, num_parallel_calls=os.cpu_count()))
    eval_dataset  = eval_dataset.prefetch(8)

    train_dataset = dataset.skip(eval_len)
    train_dataset = train_dataset.apply(tf.data.experimental.shuffle_and_repeat(6000))
    train_dataset = train_dataset.apply(tf.data.experimental.map_and_batch(_parse_proto, batch_size, num_parallel_calls=os.cpu_count()))
    train_dataset = train_dataset.prefetch(8)

    n_batches_train = math.ceil(train_len / batch_size)
    n_batches_eval = math.ceil(eval_len / batch_size)

    return train_dataset, n_batches_train, eval_dataset, n_batches_eval
