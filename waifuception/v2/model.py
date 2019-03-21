import math
import os
import os.path as osp
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from keras.models import Model
from keras.layers import Dense, Dropout, Input, Activation, BatchNormalization, Conv2D, Add, GlobalAveragePooling2D
from keras.optimizers import RMSprop
from keras import callbacks
from keras import metrics
from keras import backend as K
from keras.applications import resnet_v2

import classes as c
import classifier_metrics

def residual_block(x, filters, name, strides=(1,1), resize_shortcut=False):
    """
    A ResNet-v2 bottleneck layer.
    """
    preactivation = BatchNormalization(axis=-1, name='bn_'+name+'_preact')(x)
    preactivation = Activation('relu')(preactivation)
    blk = Conv2D(filters[0], (1, 1), kernel_initializer='he_normal', strides=strides, name='res_'+name+'_a')(preactivation)

    blk = BatchNormalization(axis=-1, name='bn_'+name+'_b')(blk)
    blk = Activation('relu')(blk)
    blk = Conv2D(filters[1], (3, 3), kernel_initializer='he_normal', padding='same', name='res_'+name+'_b')(blk)

    blk = BatchNormalization(axis=-1, name='bn_'+name+'_c')(blk)
    blk = Activation('relu')(blk)
    blk = Conv2D(filters[2], (1, 1), kernel_initializer='he_normal', name='res_'+name+'_c')(blk)

    if resize_shortcut:
        shortcut = Conv2D(filters[2], (1, 1), strides=strides, name='res_'+name+'_shortcut')(preactivation)
    else:
        shortcut = x

    blk = Add()([blk, shortcut])
    return blk

def build_model(output_labels, weights_dir, lr):
    base_model = resnet_v2.ResNet101V2(include_top=False, weights='imagenet', pooling=None)
    x = base_model.output

    x = residual_block(x, [512, 512, 2048], 'conv6_1')
    x = residual_block(x, [512, 512, 2048], 'conv6_2')
    x = residual_block(x, [512, 512, 2048], 'conv6_3')
    x = GlobalAveragePooling2D()(x)
    x = Dense(107)(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=x)

    top_weightsfile = None
    top_weightsfile_epoch = None
    for weightsfile in Path(weights_dir).iterdir():
        _, epoch, val_loss = weightsfile.stem.split('.', 2)

        epoch = int(epoch)
        val_loss = float(val_loss)

        if top_weightsfile_epoch is None or epoch > top_weightsfile_epoch:
            top_weightsfile = weightsfile
            top_weightsfile_epoch = epoch

    if top_weightsfile is not None:
        print("Resuming from epoch "+str(top_weightsfile_epoch))
        print("Weights file: "+str(top_weightsfile))
        model.load_weights(str(top_weightsfile))
    else:
        top_weightsfile_epoch = 0

    optimizer = RMSprop(lr=lr, rho=0.9, epsilon=1.0, clipvalue=2.0)
    loss = 'binary_crossentropy'

    model.compile(optimizer=optimizer, loss=loss, metrics=[
        classifier_metrics.true_positive_rate,
        classifier_metrics.true_negative_rate,
        classifier_metrics.false_positive_rate,
        classifier_metrics.false_negative_rate
    ])

    return model, top_weightsfile_epoch
