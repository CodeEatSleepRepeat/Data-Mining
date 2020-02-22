from sklearn.model_selection import train_test_split
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import numpy as np


def train_test_split_data(x_data, y_data, test_size):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=123)
    return x_train, x_test, y_train, y_test


def train_test_split_one_dataset(data_set):
    #data_set_train, data_set_test
    return


def select_only_numerical_features(data_set):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical_vars = list(data_set.select_dtypes(include=numerics).columns)
    numerical_data_set = data_set[numerical_vars]

    return numerical_data_set


def read_csv(url_path):
    collected_data = pd.read_csv(url_path, encoding='ISO-8859-1')

    return collected_data


def read_exel(url_path):
    collected_data = pd.read_excel(url_path)

    return collected_data


'''
One hot encoding naspram label encoding je izabran iz razloga sto necemo da model slucajno predpostavi da ima nekih 
veza izmedju drzava, tj ako ih prebacimo u brojcani oblik, gde su drzave oznacene sa 1,2,3... onda ce on u svom algoritmu
prepoznati 1<2<3 ... a to nama ne treba.
'''


def one_hot_encoding(data_set, encode_collumns):
    ct = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(), [0])],
        # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
        remainder='passthrough'  # Leave the rest of the columns untouched
    )

    data = ct.fit_transform(data_set)

    print(data)
    data_set = np.array(ct.fit_transform(data_set), dtype=np.float)

    return data_set


def label_encoding(data_set):
    le = LabelEncoder()

    # 2/3. FIT AND TRANSFORM
    # use df.apply() to apply le.fit_transform to all columns
    labeled_data_set = data_set.apply(le.fit_transform)

    return labeled_data_set


def label_encoding2(data_set, collumn_names):
    label_encoder = LabelEncoder()

    # Encode labels in column 'species'.
    data_set[collumn_names] = label_encoder.fit_transform(data_set[collumn_names])

    data_set[collumn_names].unique()

    return data_set

