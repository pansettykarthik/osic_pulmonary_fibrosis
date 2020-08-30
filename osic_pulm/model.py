from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from utils.metric_utils import *

def build_model(X_train, hparams):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=[len(X_train.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(hparams["LEARNING_RATE"])

    #   quantile=0.5
    #   loss=tf.keras.losses.MeanSquaredLogarithmicError()
    #   loss=lambda y,f: tilted_loss(quantile,y,f)
    #   loss=log_custom_loss

    model.compile(loss=log_custom_loss,
                  optimizer=optimizer,
                  metrics=[laplace_log_metric, 'mae', 'mse'])
    return model