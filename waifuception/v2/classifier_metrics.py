import tensorflow as tf
import keras.backend as K


def false_positive_rate(y_true, y_pred):
    in_shape = K.shape(y_pred)
    final_shape = tf.stack([in_shape[0], -1])

    y_true = K.round(K.reshape(y_true, final_shape))
    y_pred = K.round(K.sigmoid(K.reshape(y_pred, final_shape)))

    fp = K.sum(K.cast(K.greater(y_pred, y_true), 'float32'), axis=1)   # false positive
    cn = K.sum(K.cast(K.equal(y_true, 0), 'float32'), axis=1)          # condition negative

    return K.mean(fp / (cn + K.epsilon()))

def false_negative_rate(y_true, y_pred):
    in_shape = K.shape(y_pred)
    final_shape = tf.stack([in_shape[0], -1])

    y_true = K.round(K.reshape(y_true, final_shape))
    y_pred = K.round(K.sigmoid(K.reshape(y_pred, final_shape)))

    fn = K.sum(K.cast(K.greater(y_true, y_pred), 'float32'), axis=1) # false negative
    cp = K.sum(K.cast(K.equal(y_true, 1), 'float32'), axis=1)        # condition positive

    return K.mean(fn / (cp + K.epsilon()))

def true_positive_rate(y_true, y_pred):
    in_shape = K.shape(y_pred)
    final_shape = tf.stack([in_shape[0], -1])

    y_true = K.reshape(y_true, final_shape)
    y_pred = K.sigmoid(K.reshape(y_pred, final_shape))

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
    y_pred = K.sigmoid(K.reshape(y_pred, final_shape))

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
