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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)

def main():
    with tf.device('/cpu:0'):
        print("Building model...")
        model, resume_from_epoch = waifuception.build_model(None)

    with open(sys.argv[1], 'w', encoding='utf-8') as f:
        f.write(model.to_json())
    

if __name__ == '__main__':
    main()
