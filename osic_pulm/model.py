from tensorflow import keras
from tensorflow.keras import layers
from utils.metric_utils import *

def build_model(X_train, hparams):
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=[len(X_train.keys())]),
        layers.Dense(256, activation='relu'),
        layers.Dense(1)
    ])

    # optimizer = tf.keras.optimizers.RMSprop(hparams["LEARNING_RATE"])
    # loss = tf.keras.losses.MeanSquaredLogarithmicError()

    optimizer = tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999)

    model.compile(loss=log_custom_loss,
                  optimizer=optimizer,
                  metrics=[laplace_log_metric, 'mae', 'mse'])
    return model
