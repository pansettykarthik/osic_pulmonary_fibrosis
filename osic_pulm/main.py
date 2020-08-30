import os
import numpy as np
import pandas as pd
import seaborn as sns
import shutil
from datetime import datetime as dt

import tensorflow as tf
from tensorflow import keras
from model import build_model
from utils.metric_utils import laplace_log_likelihood
from utils.train_utils import *
from hyperparameters import hparams

# BASE_PATH = "/kaggle/input/osic-pulmonary-fibrosis-progression/"
BASE_PATH = ""

def train(normalized_X_train, y_train, hparams):
    model = build_model(normalized_X_train, hparams)
    print(model.summary())

    time_now = dt.now().strftime("%Y%m%d_%H%M%S")
    my_callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=BASE_PATH + 'logs/' + str(time_now)),
    ]

    history = model.fit(
        normalized_X_train, y_train, batch_size=32,
        epochs=hparams["EPOCHS"], validation_split=0.2, verbose=1, callbacks=my_callbacks)

    return model


def inference(model, X_test, y_test):
    y_pred = model.predict(X_test).reshape(-1)
    return y_pred, laplace_log_likelihood(y_test, y_pred, np.std(y_pred))


if __name__ == '__main__':
    train_csv, test_csv = load_data()
    X_train, y_train, X_test, y_test = process_data(train_csv, test_csv)
    normalized_X_train, normalized_X_test = normalize_data(X_train, X_test)

    model = train(normalized_X_train, y_train, hparams)
    y_pred, metric = inference(model, normalized_X_test, y_test.tolist())
    print(metric)
    print(np.std(y_pred))

    loss, log_laplace, mae, mse = model.evaluate(normalized_X_test, y_test, verbose=2)
