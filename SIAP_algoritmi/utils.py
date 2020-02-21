from sklearn.model_selection import train_test_split


def train_test_split_data(x_data, y_data, test_size):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=123)
    return x_train, x_test, y_train, y_test


def select_only_numerical_features(data_set):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical_vars = list(data_set.select_dtypes(include=numerics).columns)
    numerical_data_set = data_set[numerical_vars]

    return numerical_data_set
