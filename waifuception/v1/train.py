import pandas as pd
import math
import numpy as np
import tensorflow as tf
import os
import os.path as osp
import shutil
from pathlib import Path
import sys

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Activation, BatchNormalization, Conv2D, Add, GlobalAveragePooling2D
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
    [True, gender_tags,         'softmax', 'gender'],
    [True, rating_classes,      'softmax', 'rating'],
    [True, hair_color_tags,     'sigmoid', 'hair_color'],
    [True, hair_length_tags,    'softmax', 'hair_length'],
    [True, hair_style_tags,     'sigmoid', 'hair_style'],
    [True, eye_color_tags,      'sigmoid', 'eye_color'],
    [True, eye_misc_tags,       'sigmoid', 'eye_misc'],
    [True, expression_tags,     'sigmoid', 'expression'],
    [True, breast_tags,         'softmax_plus_no_class', 'breasts'],
    [True, ass_tags,            'sigmoid', 'ass'],
    [True, pose_tags,           'softmax_plus_no_class', 'pose'],
    [True, attire_tags,         'sigmoid', 'attire'],
]

df = pd.read_csv('./2018-current.csv.gz', index_col=1)

names = []
category_weights = []
all_tags = []
for active, cat, __, name in tags_list:
    if active:
        all_tags.extend(cat)

        n_samples = 0
        for tag in cat:
            n_samples += len(df.index[df[tag] == 1].tolist())

        category_weights.append(n_samples)
        names.append(name)

category_weights = np.array(category_weights) / sum(category_weights)
category_weights = 1 - category_weights

N_CLASSES = len(all_tags)
DATASET_LENGTH = 50000

weights_dir = '/mnt/data/waifuception-checkpoints-7/'

label_e = 0.1 # label smoothing epsilon value

#img_mean = np.load('/mnt/data/dataset_mean.npy')

def false_positive_rate(y_true, y_pred):
    in_shape = K.shape(y_pred)
    final_shape = tf.stack([in_shape[0], -1])

    y_true = K.reshape(y_true, final_shape)
    y_pred = K.reshape(y_pred, final_shape)

    fp = K.sum(K.cast(K.greater(K.round(y_pred), K.round(y_true)), 'float32'), axis=1) # false positive
    cn = K.sum(K.cast(K.equal(K.round(y_true), 0), 'float32'), axis=1)                 # condition negative

    return K.mean(fp / (cn + K.epsilon()))

def false_negative_rate(y_true, y_pred):
    in_shape = K.shape(y_pred)
    final_shape = tf.stack([in_shape[0], -1])

    y_true = K.reshape(y_true, final_shape)
    y_pred = K.reshape(y_pred, final_shape)

    fn = K.sum(K.cast(K.greater(K.round(y_true), K.round(y_pred)), 'float32'), axis=1) # false negative
    cp = K.sum(K.cast(K.equal(K.round(y_true), 1), 'float32'), axis=1)                 # condition positive

    return K.mean(fn / (cp + K.epsilon()))

def true_positive_rate(y_true, y_pred):
    in_shape = K.shape(y_pred)
    final_shape = tf.stack([in_shape[0], -1])

    y_true = K.reshape(y_true, final_shape)
    y_pred = K.reshape(y_pred, final_shape)

    predicted_true = K.equal(K.round(y_pred), 1)
    condition_true = K.equal(K.round(y_true), 1)

    true_positives = K.all(K.stack(
        [predicted_true,
        condition_true],
        axis=2
    ), axis=2)

    tp = K.sum(K.cast(true_positives, 'float32'), axis=1)
    cp = K.sum(K.cast(condition_true, 'float32'), axis=1)

    return K.mean(tp / cp)

def true_negative_rate(y_true, y_pred):
    in_shape = K.shape(y_pred)
    final_shape = tf.stack([in_shape[0], -1])

    y_true = K.reshape(y_true, final_shape)
    y_pred = K.reshape(y_pred, final_shape)

    predicted_false = K.equal(K.round(y_pred), 0)
    condition_false = K.equal(K.round(y_true), 0)

    true_negatives = K.all(K.stack(
        [predicted_false,
        condition_false],
        axis=2
    ), axis=2)

    tn = K.sum(K.cast(true_negatives, 'float32'), axis=1)
    cn = K.sum(K.cast(condition_false, 'float32'), axis=1)

    return K.mean(tn / cn)

class ClassifierMetrics(callbacks.Callback):
    def __init__(self, eval_dataset, steps_for_eval):
        callbacks.Callback.__init__(self)
        self.eval_ds = eval_dataset
        self.steps_for_eval = steps_for_eval

    def on_epoch_end(self, epoch, log={}):
        sess = K.get_session()
        it = self.eval_ds.make_one_shot_iterator()
        next_batch = it.get_next()

        y_preds = []
        y_trues = []

        for i in range(self.steps_for_eval):
            x_eval, y_eval = sess.run(next_batch)
            y_pred = self.model.predict(x_eval, batch_size=x_eval.shape[0])

            y_eval = np.concatenate([y_eval[name] for name in names], axis=-1)
            y_pred = np.concatenate(y_pred, axis=-1)

            y_preds.append(y_pred)
            y_trues.append(y_eval)

        y_pred = np.rint(np.concatenate(y_preds, axis=0))
        y_true = np.rint(np.concatenate(y_trues, axis=0))

        _f1 = f1_score(y_true, y_pred, average='weighted')
        _precision = precision_score(y_true, y_pred, average='weighted')
        _recall = recall_score(y_true, y_pred, average='weighted')

        print("\nF1: {:.3f} - precision: {:.3f} - recall: {:.3f}\n".format(
            _f1, _precision, _recall
        ))
        sys.stdout.flush()

def weighted_crossentropy(y_true, y_pred):
    return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, 28)

def split_by_categories(inputs):
    splits = []
    is_active = []
    names = []

    for active, cat, activation, name in tags_list:
        splits.append(len(cat))
        is_active.append(active)
        names.append(name)

    split_tensors = tf.split(inputs, splits, axis=-1)
    ret = {}

    for active, tensor, name in zip(is_active, split_tensors, names):
        if active:
            ret[name] = tensor

    return ret

def _parse_proto(example_proto):
    features = {
        'img' :    tf.FixedLenFeature((), tf.string, default_value=""),
        'labels' : tf.FixedLenFeature((133,), tf.int64, default_value=np.zeros(133)),
        'shape' :  tf.FixedLenFeature((3,), tf.int64, default_value=(299, 299, 3))
    }

    parsed_features = tf.parse_single_example(example_proto, features)

    img_decoded = tf.decode_raw(parsed_features['img'], tf.uint8)
    img_out = tf.image.convert_image_dtype(tf.reshape(img_decoded, (299, 299, 3)), tf.float32)

    img_out = tf.image.random_flip_left_right(img_out)
    img_out = tf.image.random_flip_up_down(img_out)
    #img_out = tf.contrib.image.rotate(img_out, tf.random.uniform([], 0, 2*np.pi))

    # NOTE: the pretrained models expect input image pixels to lie in the range [-1, 1]!
    img_out = (img_out - 0.5) * 2.0
    #img_out = img_out - img_mean

    true_labels = tf.cast(parsed_features['labels'], tf.float32)
    true_label_categories = split_by_categories(true_labels)
    for idx in true_label_categories.keys():
        label_tensor = true_label_categories[idx]

        cat = None
        for c in tags_list:
            if c[3] == idx:
                cat = c
                break

        if cat[2] == 'softmax_plus_no_class':
            # add an extra class for 'no tag':
            no_class = tf.reduce_all(tf.less(label_tensor, 0.5), axis=-1)
            no_class = tf.where(no_class, tf.ones_like(no_class, dtype=tf.float32), tf.zeros_like(no_class, dtype=tf.float32))
            no_class = tf.expand_dims(no_class, axis=-1)

            label_tensor = tf.concat([label_tensor, no_class], axis=-1)
            true_label_categories[idx] = label_tensor

    #train_labels = ((1.0 - label_e) * true_labels) + (label_e / N_CLASSES)

    return img_out, true_label_categories

def residual_block(x, filters, name, strides=(1,1), resize_shortcut=False):
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

def build_model(lr):
    #base_model = inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')
    #base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(299, 299, 3), pooling=None)

    x = base_model.output
    x = residual_block(x, [256, 256, 1024], 'prehead_1', resize_shortcut=True)
    x = residual_block(x, [256, 256, 1024], 'prehead_2')
    x = residual_block(x, [256, 256, 1024], 'prehead_3')
    x = residual_block(x, [256, 256, 1024], 'prehead_4')

    heads = []
    losses = {}
    metrics = {}
    for active, taglist, activation_fn, name in tags_list:
        if not active:
            continue

        out_units = len(taglist)

        if activation_fn == 'softmax_plus_no_class':
            activation_fn = 'softmax'
            out_units += 1

        head = residual_block(x, [128, 128, 512], name+'_a', resize_shortcut=True)
        head = residual_block(head, [128, 128, 512], name+'_b')
        head = residual_block(head, [128, 128, 512], name+'_c')
        head = residual_block(head, [128, 128, 512], name+'_d')

        head = GlobalAveragePooling2D()(head)
        head = Dense(out_units, activation=activation_fn, name=name)(head)

        heads.append(head)

        if activation_fn == 'softmax':
            losses[name] = 'categorical_crossentropy'
            metrics[name] = 'categorical_accuracy'
        elif activation_fn == 'sigmoid':
            losses[name] = 'binary_crossentropy'
            metrics[name] = [false_positive_rate, false_negative_rate]

    optimizer = RMSprop(lr=lr, rho=0.9, epsilon=1.0, clipvalue=2.0)

    model = Model(inputs=base_model.input, outputs=heads)

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

    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
    #model.summary()

    return model, top_weightsfile_epoch

def main():
    try:
        if osp.isdir('/mnt/data/tensorboard-logs'):
            shutil.rmtree('/mnt/data/tensorboard-logs')

        os.mkdir('/mnt/data/tensorboard-logs')
    except OSError:
        print("Warning: could not clear tensorboard data")

    try:
        if not osp.isdir(weights_dir):
            os.mkdir(weights_dir)
    except OSError:
        print("Warning: could not make checkpoints dir")

    base_lr = 0.1
    batch_size = 32

    dataset = tf.data.TFRecordDataset('./danbooru2018-preprocessed/dataset-2.tfrecords')
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    dataset = dataset.take(DATASET_LENGTH)

    eval_len = int(DATASET_LENGTH * 0.05)
    train_len = DATASET_LENGTH - eval_len

    eval_dataset  = dataset.take(eval_len)
    eval_dataset  = eval_dataset.apply(tf.data.experimental.shuffle_and_repeat(500))
    eval_dataset  = eval_dataset.apply(tf.data.experimental.map_and_batch(_parse_proto, batch_size, num_parallel_calls=os.cpu_count()))
    eval_dataset  = eval_dataset.prefetch(1)

    train_dataset = dataset.skip(eval_len)
    train_dataset = train_dataset.apply(tf.data.experimental.shuffle_and_repeat(6000))
    train_dataset = train_dataset.apply(tf.data.experimental.map_and_batch(_parse_proto, batch_size, num_parallel_calls=os.cpu_count()))
    train_dataset = train_dataset.prefetch(1)

    print("Building model...")
    model, resume_from_epoch = build_model(base_lr)

    n_batches_train = math.ceil(train_len / batch_size)
    n_batches_eval = math.ceil(eval_len / batch_size)

    print("Starting training.")
    model.fit(
        train_dataset.make_one_shot_iterator(),
        steps_per_epoch  = n_batches_train,
        validation_data  = eval_dataset.make_one_shot_iterator(),
        validation_steps = n_batches_eval,
        epochs           = 200,
        verbose          = 1,
        initial_epoch    = resume_from_epoch,
        callbacks=[
            callbacks.ModelCheckpoint(weights_dir+'weights.{epoch:03d}.{val_loss:.04f}.hdf5'),
            #callbacks.LearningRateScheduler(lambda epoch, cur_lr: base_lr * np.power(0.87, epoch)),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001),
            callbacks.TensorBoard('/mnt/data/tensorboard-logs', update_freq=700),
            #ClassifierMetrics(eval_dataset, n_batches_eval)
        ]
    )


if __name__ == '__main__':
    main()
