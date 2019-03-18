import pandas as pd
import numpy as np
from pathlib import Path
import sys
from PIL import Image

import tensorflow as tf
    
base_path = Path('G:/danbooru2018/original')
shape_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[299, 299, 3]))

def main():
    print("Loading metadata...")
    df = pd.read_csv('./2018-current.csv.gz', index_col=0)
    
    n = 0
    with tf.python_io.TFRecordWriter('./dataset-2.tfrecords') as writer:
        for idx, row in df.iterrows():
            try:
                with Image.open(base_path / row[2]) as im:
                    resized = im.convert('RGB').resize((299, 299))
                    
                    features = {
                        'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.asarray(resized, np.uint8).tobytes()])),
                        'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=row[3:])),
                        'shape': shape_feature
                    }
                
                example_proto = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example_proto.SerializeToString())
            except OSError:
                pass

            #n += 1
            #if n % 100 == 0:
                #print("Processed image {} of {}".format(n, n_total))


if __name__ == '__main__':
    main()
