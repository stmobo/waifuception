import math
import os
import os.path as osp
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Activation, BatchNormalization, Conv2D, Add, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import callbacks
from tensorflow.keras import metrics
from tensorflow.keras import backend as K

from . import classes as c


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


def conditional_categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred) * 
