import pandas as pd
import math
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import inception_v3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks
from tensorflow.keras import metrics
from tensorflow.keras import backend as K

gender_tags = ['female', 'male']
rating_classes = ['safe', 'questionable', 'explicit']

hair_color_tags = ['aqua_hair', 'black_hair', 'blonde_hair', 'blue_hair', 'brown_hair', 'green_hair', 'grey_hair', 'orange_hair', 'pink_hair', 'purple_hair', 'red_hair', 'silver_hair', 'white_hair', 'multicolored_hair']
hair_length_tags = ['very_short_hair', 'short_hair', 'medium_hair', 'long_hair', 'very_long_hair', 'bald']
hair_style_tags = ['curly_hair', 'drill_hair', 'flipped_hair', 'hair_flaps', 'messy_hair', 'pointy_hair', 'ringlets', 'spiked_hair', 'wavy_hair', 'bangs', 'ahoge', 'braid', 'hair_bun', 'ponytail', 'twintails']

eye_color_tags = ['aqua_eyes', 'black_eyes', 'blue_eyes', 'brown_eyes', 'green_eyes', 'grey_eyes', 'orange_eyes', 'pink_eyes', 'purple_eyes', 'red_eyes', 'silver_eyes', 'white_eyes', 'yellow_eyes', 'heterochromia']
eye_misc_tags = ['closed_eyes', 'one_eye_closed', 'glasses']

expression_tags = ['angry', 'annoyed', 'blush', 'embarrassed', 'bored', 'confused', 'crazy', 'expressionless', 'happy', 'nervous', 'pout', 'sad', 'scared', 'worried', 'serious', 'sigh', 'sleepy', 'thinking', 'ahegao']

breast_tags = ['flat_chest', 'small_breasts', 'medium_breasts', 'large_breasts', 'huge_breasts', 'gigantic_breasts']
ass_tags = ['ass', 'flat_ass', 'huge_ass']
pose_tags = ['kneeling', 'lying', 'sitting', 'standing']

attire_tags = [
    'hat',
    'hairband',
    'hair_bow',
    'hair_ribbon',
    'helmet',
    'shirt',
    'dress',
    'jacket',
    'vest',
    'pants',
    'shorts',
    'skirt',
    'jeans',
    'pantyhose',
    'socks',
    'thighhighs',
    'shoes',
    'boots',
    'sandals',
    'uniform',
    'apron',
    'armor',
    'cape',
    'hood',
    'school_uniform',
    'serafuku',
    'suit',
    'swimsuit',
    'bikini',
    'japanese_clothes',
    'kimono',
    'earrings',
    'hair_ornament',
    'hairclip',
    'detached_sleeves',
    'gloves',
    'fingerless_gloves',
    'elbow_gloves',
    'belt',
    'headphones_around_neck',
    'goggles_around_neck',
    'necklace',
    'necktie',
    'scarf'
]

# list of tag categories, in order
tags_list = [
    gender_tags,       
    rating_classes,    
    hair_color_tags,   
    hair_length_tags,  
    hair_style_tags,   
    eye_color_tags,    
    eye_misc_tags,     
    expression_tags,   
    breast_tags,       
    ass_tags,          
    pose_tags,         
    attire_tags,       
]

N_CLASSES = sum([len(cat) for cat in tags_list])
DATASET_LENGTH = 301785
label_e = 0.1 # label smoothing epsilon value

img_mean = np.load('/mnt/data/dataset_mean.npy')

def subset_accuracy_score(y_true, y_pred):
    differing_labels = K.sum(K.abs(y_true - K.round(y_pred)), axis=1)
    return K.mean(K.equal(differing_labels, 0))

def n_differing_labels(y_true, y_pred):
    differing_labels = K.sum(K.abs(y_true - K.round(y_pred)), axis=1)
    return K.mean(differing_labels)

def _parse_proto(example_proto):
    features = {
        'img' :    tf.FixedLenFeature((), tf.string, default_value=""),
        'labels' : tf.FixedLenFeature((133,), tf.int64, default_value=np.zeros(133)),
        'shape' :  tf.FixedLenFeature((3,), tf.int64, default_value=(299, 299, 3))
    }
    
    parsed_features = tf.parse_single_example(example_proto, features)
    
    img_decoded = tf.decode_raw(parsed_features['img'], tf.uint8)
    img_out = tf.image.convert_image_dtype(tf.reshape(img_decoded, (299, 299, 3)), tf.float32)
    
    #img_out = img_out - img_mean
    
    #img_out = tf.image.random_flip_left_right(img_out)
    #img_out = tf.image.random_flip_up_down(img_out)
    #img_out = tf.contrib.image.rotate(img_out, tf.random.uniform([], 0, 2*np.pi))
    
    # NOTE: the pretrained models expect input image pixels to lie in the range [-1, 1]!
    img_out = (img_out - 0.5) * 2.0
    
    true_labels = tf.cast(parsed_features['labels'], tf.float32)
    train_labels = ((1.0 - label_e) * true_labels) + (label_e / N_CLASSES)
    
    return img_out, train_labels

def build_model(lr):
    base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')

    x = base_model.output
    x = Dropout(0.20)(x)
    predictions = Dense(N_CLASSES, activation='sigmoid')(x)

    #for layer in base_model.layers:
    #    layer.trainable = False

    optimizer = RMSprop(lr=lr, decay=0.9, epsilon=1.0, clipvalue=2.0)

    model = Model(inputs=base_model.input, outputs=predictions)
    #model.load_weights('/mnt/data/waifuception-checkpoints-2/weights.027-0.4452.hdf5')
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[subset_accuracy_score, n_differing_labels])
    #model.summary()

    return model

def main():
    base_lr = 0.045
    
    dataset = tf.data.TFRecordDataset('/mnt/data/danbooru2018-preprocessed/dataset-2.tfrecords')
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    
    eval_len = int(DATASET_LENGTH * 0.1)
    train_len = DATASET_LENGTH - eval_len
    
    eval_dataset  = dataset.take(eval_len)
    eval_dataset  = eval_dataset.apply(tf.data.experimental.shuffle_and_repeat(500))
    eval_dataset  = eval_dataset.apply(tf.data.experimental.map_and_batch(_parse_proto, 32, num_parallel_calls=4))
    eval_dataset  = eval_dataset.prefetch(1)
    
    train_dataset = dataset.skip(eval_len)
    train_dataset = train_dataset.apply(tf.data.experimental.shuffle_and_repeat(500))
    train_dataset = train_dataset.apply(tf.data.experimental.map_and_batch(_parse_proto, 32, num_parallel_calls=4))
    train_dataset = train_dataset.prefetch(1)
    
    print("Building model...")
    model = build_model(base_lr)

    n_batches_train = math.ceil(train_len / 32) - 1
    n_batches_eval = math.ceil(eval_len / 32) - 1
    
    print("Starting training.")
    model.fit(
        train_dataset.make_one_shot_iterator(),
        steps_per_epoch  = n_batches_train,
        validation_data  = eval_dataset.make_one_shot_iterator(),
        validation_steps = n_batches_eval,
        epochs           = 200,
        verbose          = 1,
        initial_epoch    = 0,
        callbacks=[
            callbacks.ModelCheckpoint('/mnt/data/waifuception-checkpoints-3/weights.{epoch:03d}-{val_loss:.4f}.hdf5'),
            callbacks.LearningRateScheduler(lambda epoch, cur_lr: base_lr * np.power(0.94, (epoch+1) // 2)),
            callbacks.TensorBoard('/mnt/data/tensorboard-logs')
        ]
    )
    

if __name__ == '__main__':
    main()
