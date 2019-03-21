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
from keras.utils import multi_gpu_model

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

def build_model(weights_dir):
    base_model = resnet_v2.ResNet101V2(include_top=False, weights='imagenet', pooling=None)
    x = base_model.output

    x = residual_block(x, [512, 512, 2048], 'conv6_1')
    x = residual_block(x, [512, 512, 2048], 'conv6_2')
    x = residual_block(x, [512, 512, 2048], 'conv6_3')
    x = GlobalAveragePooling2D()(x)
    x = Dense(106)(x)

    model = Model(inputs=base_model.input, outputs=x)

    if weights_dir is not None:
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
    else:
        top_weightsfile_epoch = 0

    return model, top_weightsfile_epoch

def weighted_crossentropy(y_true, y_pred):
    return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, 13 * 1.5)

def compile_model(model, lr):
    optimizer = RMSprop(lr=lr, rho=0.9, epsilon=1.0, clipvalue=2.0)

    model.compile(optimizer=optimizer, loss=weighted_crossentropy, metrics=[
        classifier_metrics.true_positive_rate,
        classifier_metrics.true_negative_rate,
        classifier_metrics.false_positive_rate,
        classifier_metrics.false_negative_rate
    ])

class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)
