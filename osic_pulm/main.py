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
CONFIDENCE = 250

def train(normalized_X_train, y_train, hparams):
    model = build_model(normalized_X_train, hparams)
    print(model.summary())

    time_now = dt.now().strftime("%Y%m%d_%H%M%S")
    my_callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=BASE_PATH + 'logs/' + str(time_now)),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                             patience=10, verbose=1, min_lr=1e-8)
    ]

    history = model.fit(
        normalized_X_train, y_train, batch_size=32,
        epochs=hparams["EPOCHS"], validation_split=0.2, verbose=1, callbacks=my_callbacks)

    return model


def validate(model, X_test, y_test):
    y_pred = model.predict(X_test).reshape(-1)
    return y_pred, laplace_log_likelihood(y_test, y_pred, CONFIDENCE)


def predict_test(model):
    test_df, submit_df = load_test()
    normalized_test_df = process_test(test_df, X_train, normalized_X_train)
    y_test_df = model.predict(normalized_test_df).reshape(-1)
    # print(y_test_df)

    submit_df['Patient_Week'] = test_df['Patient_Week']
    submit_df['FVC'] = y_test_df
    submit_df['Confidence'] = CONFIDENCE

    return submit_df


if __name__ == '__main__':
    train_csv, test_csv = load_data()
    X_train, y_train, X_test, y_test = process_data(train_csv, test_csv)
    normalized_X_train, normalized_X_test = normalize_data(X_train, X_test)

    model = train(normalized_X_train, y_train, hparams)
    y_pred, metric = validate(model, normalized_X_test, y_test.tolist())
    # print(y_pred)
    print("Laplace log likelihood: ", metric)
    print("Confidence: ", np.std(y_pred))

    # loss, log_laplace, mae, mse = model.evaluate(normalized_X_test, y_test, verbose=2)

    # # TRAIN ON WHOLE DATA AFTER FINDING BEST PARAMS
    # X_train_df = pd.concat([normalized_X_train, normalized_X_test])
    # y_train_df = pd.concat([y_train, y_test])
    # model = train(X_train_df, y_train_df, hparams)

    submit_df = predict_test(model)
    print("\nPREDICTIONS:\n", submit_df)

    # submit_df.to_csv("submission.csv", sep=",", index=False)
