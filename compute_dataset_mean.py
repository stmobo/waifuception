import math
import numpy as np
import tensorflow as tf

def _parse_proto(example_proto):
    features = {
        'img' :    tf.FixedLenFeature((), tf.string, default_value=""),
        'labels' : tf.FixedLenFeature((133,), tf.int64, default_value=np.zeros(133)),
        'shape' :  tf.FixedLenFeature((3,), tf.int64, default_value=(299, 299, 3))
    }
    
    parsed_features = tf.parse_single_example(example_proto, features)
    
    img_decoded = tf.decode_raw(parsed_features['img'], tf.uint8)
    img_decoded = tf.reshape(img_decoded, (299, 299, 3))
    img_out = tf.image.convert_image_dtype(img_decoded, tf.float32)

    return img_out

def main():    
    dataset = tf.data.TFRecordDataset('/mnt/data/danbooru2018-preprocessed/dataset-2.tfrecords')
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    dataset = dataset.map(_parse_proto, num_parallel_calls=16)
    dataset = dataset.prefetch(2)
    
    iterator = dataset.make_one_shot_iterator()
    next_input = iterator.get_next()
    
    n = tf.Variable(0, dtype=tf.float32)
    cur_mean = tf.get_variable('image_mean', (299, 299, 3), tf.float32)
    
    init_mean = tf.assign(cur_mean, np.zeros(shape=(299, 299, 3), dtype=np.float32))
    init_n = tf.assign(n, 0)
    
    next_n = tf.assign(n, n+1)
    next_mean = tf.assign(cur_mean, cur_mean + ((next_input - cur_mean) / next_n))
    
    print("Setting up session...")
    
    with tf.Session() as sess:
        print("Initializing...")
        sess.run([init_mean, init_n])
        
        dataset_mean = np.zeros(shape=(299, 299, 3), dtype=np.float32)
        while True:
            try:
                dataset_mean, cur_idx = sess.run([next_mean, n])
                
                if int(cur_idx) % 500 == 0:
                    print("Processed {} images".format(int(cur_idx)))
                    
                    if int(cur_idx) % 10000 == 0:
                        np.save("/mnt/data/dataset_mean.npy", dataset_mean)
            except tf.errors.OutOfRangeError:
                print("Mean computation complete.")
                break

        np.save("/mnt/data/dataset_mean.npy", dataset_mean)

if __name__ == '__main__':
    main()
