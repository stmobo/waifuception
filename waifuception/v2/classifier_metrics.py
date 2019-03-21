import tensorflow as tf
import keras.backend as K


def false_positive_rate(y_true, y_pred):
    in_shape = K.shape(y_pred)
    final_shape = tf.stack([in_shape[0], -1])

    y_true = K.round(K.reshape(y_true, final_shape))
    y_pred = K.round(K.sigmoid(K.reshape(y_pred, final_shape)))

    fp = K.sum(K.cast(K.greater(y_pred, y_true), 'float32'), axis=0)   # false positive
    cn = K.sum(K.cast(K.equal(y_true, 0), 'float32'), axis=0)          # condition negative

    fpr = fp / (cn + K.epsilon())
    n_cond_negative = K.sum(K.cast(K.greater(cn, 0), 'float32'))

    return K.sum(fpr) / n_cond_negative

def false_negative_rate(y_true, y_pred):
    in_shape = K.shape(y_pred)
    final_shape = tf.stack([in_shape[0], -1])

    y_true = K.round(K.reshape(y_true, final_shape))
    y_pred = K.round(K.sigmoid(K.reshape(y_pred, final_shape)))

    fn = K.sum(K.cast(K.greater(y_true, y_pred), 'float32'), axis=0) # false negative
    cp = K.sum(K.cast(K.equal(y_true, 1), 'float32'), axis=0)        # condition positive

    fnr = fn / (cp + K.epsilon())
    n_cond_positive = K.sum(K.cast(K.greater(cp, 0), 'float32'))

    return K.sum(fnr) / n_cond_positive

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

    tp = K.sum(K.cast(true_positives, 'float32'), axis=0)
    cp = K.sum(K.cast(condition_true, 'float32'), axis=0)

    tpr = tp / (cp + K.epsilon())
    n_cond_positive = K.sum(K.cast(K.greater(cp, 0), 'float32'))

    return K.sum(tpr) / n_cond_positive

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

    tn = K.sum(K.cast(true_negatives, 'float32'), axis=0)
    cn = K.sum(K.cast(condition_false, 'float32'), axis=0)

    fpr = tn / (cn + K.epsilon())
    n_cond_negative = K.sum(K.cast(K.greater(cn, 0), 'float32'))

    return K.sum(fpr) / n_cond_negative
