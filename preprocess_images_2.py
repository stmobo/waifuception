import math
import numpy as np
import tensorflow as tf

def main():
    dataset = tf.contrib.data.CsvDataset('/mnt/data/2018-current.csv', [tf.int64, tf.string, tf.string] + [tf.float32] * 133)
    
