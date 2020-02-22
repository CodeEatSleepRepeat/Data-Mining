import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, f_regression, SelectFromModel
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


def missing_value_ratio(data_set, threshold):
    missing_values = data_set.isnull().sum() / len(data_set) * 100
    variables = data_set.columns
    variable = []
    for i in range(0, len(variables)):
        if missing_values[i] <= threshold:  # setting the threshold as 20%
            variable.append(variables[i])

    return variable


def low_variance_filter(data_set, threshold):
    var = data_set.var()
    variables = data_set.columns
    variable = []
    for i in range(0, len(var)):
        if var[i] >= threshold:  # setting the threshold as %
            variable.append(variables[i + 1])

    return variable


def random_forest(data_set, y_values, want_graph, random_state, max_depth):
    model = RandomForestRegressor(random_state=random_state, max_depth=max_depth)
    #ovde radi one hot encoding
    data_set = pd.get_dummies(data_set)
    model.fit(data_set, y_values)

    if want_graph:
        features = data_set.columns
        importances = model.feature_importances_
        indices = np.argsort(importances)
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()

    feature = SelectFromModel(model)
    fit = feature.fit_transform(data_set, y_values)

    return fit


def principal_component_analysis_pca(data_set, number_of_components):
    pca = PCA(n_components=number_of_components)
    pca_result = pca.fit_transform(data_set.values)

    return pca_result


def factor_analysis(data_set, number_of_components, want_graph):
    FA = FactorAnalysis(n_components=number_of_components).fit_transform(data_set.values)

    if want_graph:
        plt.figure(figsize=(12, 8))
        plt.title('Factor Analysis Components')
        for i in range(0, number_of_components):
            if i == (number_of_components - 1):
                plt.scatter(FA[:, i], FA[:.0])
            else:
                plt.scatter(FA[:, i], FA[:, i + 1])

    return FA


# OTHER ALGORITHMS

def backward_feature_selection(data_set, y_values, number_of_features):
    l_reg = LinearRegression()
    rfe = RFE(l_reg, number_of_features)
    rfe = rfe.fit_transform(data_set, y_values)

    return rfe


def forward_feature_selection(data_set, y_values, f_value):
    ffs = f_regression(data_set, y_values)
    variable = []
    for i in range(0, len(data_set.columns) - 1):
        if ffs[0][i] >= 10:
            variable.append(data_set.columns[i])

    return variable


def autoencoder_dimension_reduction(data_set, number_of_features):
    encoding_dim = number_of_features

    input_df = Input(shape=(171,))
    encoded = Dense(encoding_dim, activation='relu')(input_df)
    encoder = Model(input_df, encoded)
    encoded_data_set = encoder.predict(data_set)

    return encoded_data_set


def linear_discriminant_analysis(data_set, y_values, number_of_features):
    lda = LinearDiscriminantAnalysis(n_components=number_of_features)
    x_lda = lda.fit(data_set, y_values).transform(data_set)

    return x_lda


def idepedent_component_analysis_ica(data_set, number_of_components, want_graph):
    ica = FastICA(n_components=number_of_components, random_state=12)
    reduced_dataset = ica.fit_transform(data_set.values)

    if want_graph:
        plt.figure(figsize=(12, 8))
        plt.title('ICA Components')
        for i in range(0, number_of_components):
            if i == (number_of_components - 1):
                plt.scatter(reduced_dataset[:, i], reduced_dataset[:.0])
            else:
                plt.scatter(reduced_dataset[:, i], reduced_dataset[:, i + 1])

    return reduced_dataset


'''
    pip install umap-learn

    n_neighbors determines the number of neighboring points used (5 is used in paper)
    min_dist controls how tightly embedding is allowed. Larger values ensure embedded points are more evenly distributed (0.3 is used in paper)

'''


def uniform_manifold_approximation_and_projection(data_set, n_neighbors, min_dist, number_of_components, want_graph):
    umap_data = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=number_of_components).fit_transform(
        data_set.values)

    if want_graph:
        plt.figure(figsize=(12, 8))
        plt.title('Decomposition using UMAP')
        for i in range(0, number_of_components):
            if i == (number_of_components - 1):
                plt.scatter(umap_data[:, i], umap_data[:.0])
            else:
                plt.scatter(umap_data[:, i], umap_data[:, i + 1])

    return umap_data


def lasso_regularisation_reduction_dimensionality(data_set, y_values, want_to_print_values):
    # Scaling the data, as linear models benefits from feature scaling
    scaler = StandardScaler()
    scaler.fit(data_set.fillna(0))

    sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1'))
    sel_.fit(scaler.transform(data_set.fillna(0)), y_values)

    if want_to_print_values:
        sel_.get_support()

    data_set_selected = sel_.transform(data_set.fillna(0))

    return data_set_selected


def sequential_feature_selection(data_set, y_values, want_graph):
    lr = LinearRegression()
    sfs = SFS(lr,
              k_features=13,
              forward=True,
              floating=False,
              scoring='neg_mean_squared_error',
              cv=10)
    sfs = sfs.fit(data_set, y_values)
    if want_graph:
        fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')
        plt.title('Sequential Forward Selection (w. StdErr)')
        plt.grid()
        plt.show()

    return sfs
