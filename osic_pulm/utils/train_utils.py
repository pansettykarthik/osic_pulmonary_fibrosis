import pandas as pd

# DATA_PATH = "/kaggle/input/osic-pulmonary-fibrosis-progression/"
DATA_PATH = "data/"

def load_data():
    train_csv = pd.read_csv(DATA_PATH + "train.csv")
    test_csv = pd.read_csv(DATA_PATH + "test.csv")

    return train_csv, test_csv


def add_base_weeks(train_csv):
    train_csv['Patient_Week'] = train_csv['Patient'].astype(str) + '_' + train_csv['Weeks'].astype(str)

    train_csv['Weeks'] = train_csv['Weeks'].astype(int)
    train_csv['base_week'] = train_csv.groupby('Patient')['Weeks'].transform('min')
    train_csv['weeks_passed'] = train_csv['Weeks'] - train_csv['base_week']

    return train_csv


def add_base_data(train_csv):
    train_csv = add_base_weeks(train_csv)

    train_csv['base_fvc'] = 0
    train_csv['base_percent'] = 0
    for index in range(len(train_csv)):
        base_data = train_csv[train_csv['Patient'] == train_csv.iloc[index]['Patient']].iloc[0]
        train_csv.loc[index, 'base_fvc'] = base_data['FVC']
        train_csv.loc[index, 'base_percent'] = base_data['Percent']

    return train_csv


def process_categorical_data(train_csv):
    X_data = train_csv[['Sex', 'SmokingStatus']]
    X_data = pd.get_dummies(data=X_data, drop_first=True)
    selected_columns = ['Age', 'weeks_passed', 'base_fvc', 'base_percent']
    X_data[selected_columns] = train_csv[selected_columns]
    y_data = train_csv['FVC']

    return X_data, y_data

# THIS IS NOT ACTUALLY TEST SET - THIS IS JUST VALIDATION SET
def test_train_split(X_data, y_data):
    X_train = X_data.sample(frac=0.8, random_state=0)
    X_test = X_data.drop(X_train.index)

    y_train = y_data.sample(frac=0.8, random_state=0).astype(float)
    y_test = y_data.drop(y_train.index).astype(float)

    return X_train, y_train, X_test, y_test

# NOT USING TEST CSV AS OF NOW
def process_data(train_csv, test_csv):
    train_csv = add_base_data(train_csv)
    X_data, y_data = process_categorical_data(train_csv)
    X_train, y_train, X_test, y_test = test_train_split(X_data, y_data)

    return X_train, y_train, X_test, y_test


def norm(x, train_stats):
    return (x - train_stats['mean']) / train_stats['std']


def normalize_data(X_train, X_test):
    train_stats = X_train.describe().transpose()
    normalized_X_train = norm(X_train, train_stats)
    normalized_X_test = norm(X_test, train_stats)

    return normalized_X_train, normalized_X_test
