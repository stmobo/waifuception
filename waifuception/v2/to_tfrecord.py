import pandas as pd
import numpy as np
from pathlib import Path
import sys
from PIL import Image

import tensorflow as tf

base_path = Path('/mnt/data/dl/danbooru2018/original')

def main():
    have_ids = []
    fnames = {}
    jpg_fnames = {}
    for p in filter(lambda d: d.is_dir(), base_path.iterdir()):
        for f in filter(lambda p: p.is_file() and p.suffix in ['.png', '.jpg', '.jpeg'], p.iterdir()):
            iid = int(f.stem)
            have_ids.append(iid)
            fnames[iid] = f.parent.name + '/' + f.name

    print("Loading metadata...")
    df = pd.read_csv(sys.argv[1], index_col=0)

    print("Total dataset length: {}".format(len(df)))

    if len(sys.argv) > 3:
        n_samples = int(sys.argv[3])
        df = df.sample(n_samples)

    print(df.head())

    n = 0
    with tf.python_io.TFRecordWriter(sys.argv[2]) as writer:
        for idx, row in df.iterrows():
            try:
                fname = fnames[row['id']]
                labels = row[1:]

                features = {
                    'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[fname.encode('utf-8')])),
                    'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
                }

                example_proto = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example_proto.SerializeToString())
            except (OSError, KeyError):
                pass

            n += 1
            if n % 1000 == 0:
                print("Processed {} rows...".format(n))

    print("Output {} records.".format(n))

if __name__ == '__main__':
    main()
