import pandas as pd
import numpy as np
from pathlib import Path
import sys
from PIL import Image

import tensorflow as tf
import multiprocessing as mp
import queue
import time
    
base_path = Path('/mnt/data/dl/danbooru2018/original')
out_path = Path('/mnt/data/danbooru2018-preprocessed/images')
shape_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[299, 299, 3]))

def reader_worker(worker_idx, row_queue, proto_queue):    
    print("[reader-{}] Starting.".format(worker_idx))
    
    while True:
        row = row_queue.get()
        try:
            with Image.open(base_path / row[2]) as im:
                resized = im.convert('RGB').resize((299, 299))

                features = {
                    'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.asarray(resized, np.uint8).tobytes()])),
                    'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=row[3:])),
                    'shape': shape_feature
                }
            
            example_proto = tf.train.Example(features=tf.train.Features(feature=features))
            proto_queue.put(example_proto.SerializeToString())
        except OSError:
            pass
        
        row_queue.task_done()

def writer_worker(proto_queue, n_total):
    print("[writer] Starting.")
    
    n = 0
    with tf.python_io.TFRecordWriter('/mnt/data/danbooru2018-preprocessed/dataset-2.tfrecords') as writer:
        while True:
            try:
                writer.write(proto_queue.get(True, 15))
            except queue.Empty:
                print("[writer] Queue read timed out after processing {} out of {} images, exiting".format(n, n_total))
                return
                
            #writer.flush()
            proto_queue.task_done()
            
            n += 1
            if n % 100 == 0:
                print("[writer] Processed image {} of {}".format(n, n_total))
            
            if n >= (n_total-1):
                print("[writer] Processed {} out of {} images, exiting".format(n, n_total))
                return

def main():
    print("Loading metadata...")
    df = pd.read_csv('./2018-current.csv.gz', index_col=0)

    print("Beginning dataset preprocessing...")
    row_queue = mp.JoinableQueue()
    proto_queue = mp.JoinableQueue()
    
    readers = []
    
    for i in range(int(mp.cpu_count()*2)):
        w = mp.Process(target=reader_worker, args=(i, row_queue, proto_queue))
        w.daemon = True
        w.start()
        readers.append(w)
    
    writer = mp.Process(target=writer_worker, args=(proto_queue, len(df)))
    writer.daemon = True
    writer.start()
    
    for idx, row in df.iterrows():
        row_queue.put_nowait(list(row))
    
    print("[main] Enqueued all rows. Joining workers.")
    
    row_queue.join()
    for r in readers:
        r.terminate()
        r.join()
        
    print("[main] All workers exited. Joining writer.")
        
    proto_queue.join()
    #writer.terminate()
    writer.join()
    
if __name__ == '__main__':
    main()
