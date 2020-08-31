import numpy as np
import tensorflow as tf
import keras.backend as K
CONFIDENCE = 300

def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence, return_values=False):
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = - np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)

    if return_values:
        return metric
    else:
        return np.mean(metric)


def log_custom_loss(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)

    fvc_pred = y_pred
    # sigma = tf.math.reduce_std(y_pred)

    sigma = tf.constant(CONFIDENCE, dtype=tf.float32)
    sigma_clip = tf.maximum(sigma, 70)
    delta = tf.abs(y_true - fvc_pred)
    delta = tf.minimum(delta, 1000)
    sq2 = tf.sqrt(tf.dtypes.cast(2, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return K.mean(metric)


def laplace_log_metric(y_true, y_pred):
    return -log_custom_loss(y_true, y_pred)


def tilted_loss(q, y, f):
    e = (y - f)
    return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)
