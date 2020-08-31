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

def map_categorical(data):
    sex_map = {"Male": 1, "Female": 0}
    data['ex_smoker'] = data['SmokingStatus'].map(lambda x: 1 if (x == "Ex-smoker") else 0)
    data['never_smoked'] = data['SmokingStatus'].map(lambda x: 1 if (x == "Never smoked") else 0)
    data['Sex'] = data['Sex'].map(sex_map)

    return data

def process_categorical_data(train_csv):


    X_data = train_csv[['SmokingStatus']]
    selected_columns = ['Age', 'weeks_passed', 'base_fvc', 'base_percent', 'Sex', 'base_week']
    X_data[selected_columns] = train_csv[selected_columns]
    y_data = train_csv['FVC']
    X_data = map_categorical(X_data)

    X_data['base_percent'] /= 100.
    X_data['ref_fvc'] = X_data['base_fvc'] / X_data['base_percent']

    X_data = X_data.drop(columns=['SmokingStatus'])

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
    categorical_columns = ['Sex', 'ex_smoker', 'never_smoked']
    continuous_columns = ['Age', 'weeks_passed', 'ref_fvc', 'base_fvc', 'base_percent', 'base_week']
    normalized_X_train = norm(X_train[continuous_columns], train_stats)
    normalized_X_train[categorical_columns] = X_train[categorical_columns]
    normalized_X_test = norm(X_test[continuous_columns], train_stats)
    normalized_X_test[categorical_columns] = X_test[categorical_columns]

    normalized_X_train = norm(X_train, train_stats)
    normalized_X_test = norm(X_test, train_stats)

    return normalized_X_train, normalized_X_test


def load_test():
    test_df = pd.read_csv(DATA_PATH + 'test.csv')
    submit_df = pd.read_csv(DATA_PATH + 'sample_submission.csv')

    test_df = test_df.rename(columns={'Weeks': 'base_week', 'FVC': 'base_fvc', 'Percent': 'base_percent'})

    submit_df['Patient'] = submit_df.Patient_Week.str.split('_').str[0]
    submit_df['Weeks'] = submit_df.Patient_Week.str.split('_').str[1].astype(int)

    test_df = test_df.merge(submit_df, on='Patient')
    test_df['weeks_passed'] = test_df['Weeks'] - test_df['base_week']
    test_df['base_percent'] /= 100.
    test_df['ref_fvc'] = test_df['base_fvc'] / test_df['base_percent']
    test_df = test_df.set_index('Patient_Week')
    return test_df, submit_df[['Patient_Week', 'FVC', 'Confidence']]


def process_test(test_df):
    test_df = map_categorical(test_df)
    test_df = test_df.drop(columns=['SmokingStatus', 'Confidence', 'FVC', 'Weeks'])

    test_stats = test_df.describe().transpose()
    categorical_columns = ['Sex', 'ex_smoker', 'never_smoked']
    continuous_columns = ['Age', 'weeks_passed', 'ref_fvc', 'base_fvc', 'base_percent', 'base_week']

    def norm(x, train_stats):
        return (x - train_stats['mean']) / train_stats['std']

    normalized_test_df = norm(test_df[continuous_columns], test_stats)
    normalized_test_df[categorical_columns] = test_df[categorical_columns]

    return normalized_test_df
