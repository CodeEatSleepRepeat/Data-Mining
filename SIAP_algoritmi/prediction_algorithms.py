from sys import stdout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from lifelines import CoxPHFitter
from sklearn.linear_model import ElasticNet, ElasticNetCV
import sklearn.metrics


# import autosklearn.regression


# PLS
def partial_least_squares(X, y, n_comp, plot_components=True):
    '''Run PLS including a variable number of components, up to n_comp,
       and calculate MSE '''
    mse = []
    component = np.arange(1, n_comp)
    for i in component:
        pls = PLSRegression(n_components=i)
        # Cross-validation
        y_cv = cross_val_predict(pls, X, y, cv=10)
        mse.append(mean_squared_error(y, y_cv))
        comp = 100 * (i + 1) / 40
        # Trick to update status on the same line
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")
    # Calculate and print the position of minimum in MSE
    msemin = np.argmin(mse)
    print("Suggested number of components: ", msemin + 1)
    stdout.write("\n")
    if plot_components is True:
        with plt.style.context(('ggplot')):
            plt.plot(component, np.array(mse), '-v', color='blue', mfc='blue')
            plt.plot(component[msemin], np.array(mse)[msemin], 'P', ms=10, mfc='red')
            plt.xlabel('Number of PLS components')
            plt.ylabel('MSE')
            plt.title('PLS')
            plt.xlim(left=-1)
        plt.show()
    # Define PLS object with optimal number of components
    pls_opt = PLSRegression(n_components=msemin + 1)
    # Fir to the entire dataset
    pls_opt.fit(X, y)
    y_c = pls_opt.predict(X)
    # Cross-validation
    y_cv = cross_val_predict(pls_opt, X, y, cv=10)
    # Calculate scores for calibration and cross-validation
    score_c = r2_score(y, y_c)
    score_cv = r2_score(y, y_cv)
    # Calculate mean squared error for calibration and cross validation
    mse_c = mean_squared_error(y, y_c)
    mse_cv = mean_squared_error(y, y_cv)
    print('R2 calib: %5.3f' % score_c)
    print('R2 CV: %5.3f' % score_cv)
    print('MSE calib: %5.3f' % mse_c)
    print('MSE CV: %5.3f' % mse_cv)
    # Plot regression and figures of merit
    rangey = max(y) - min(y)
    rangex = max(y_c) - min(y_c)
    # Fit a line to the CV vs response
    z = np.polyfit(y, y_c, 1)
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(y_c, y, c='red', edgecolors='k')
        # Plot the best fit line
        ax.plot(np.polyval(z, y), y, c='blue', linewidth=1)
        # Plot the ideal 1:1 line
        ax.plot(y, y, color='green', linewidth=1)
        plt.title('$R^{2}$ (CV): ' + str(score_cv))
        plt.xlabel('Predicted $^{\circ}$Brix')
        plt.ylabel('Measured $^{\circ}$Brix')
        plt.show()
    return


# zahteva numericke vrednosti
def XGBoost_regression(data_set, y_values):
    xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                              max_depth=5, alpha=10, n_estimators=10)
    xg_reg.fit(data_set, y_values)

    return xg_reg


def cox_regression(data_set, duration_collumn, y_collumn_name):
    cph = CoxPHFitter()
    cph.fit(data_set, duration_collumn, event_col=y_collumn_name)

    return cph


def elastic_regression(data_set, y_value, alpha=0.01):
    elastic = ElasticNet(alpha=alpha).fit(data_set, y_value)

    return elastic


'''
Za svaki feature moramo navesti kojeg je on tipa npr:
feature_types = (['numerical'] * 3) + ['categorical'] + (['numerical'] * 9)   
kazemo ima prva 3 featurea tipa numerical pa ide jedan categorical pa 0 numerical posle njega

def auto_sclearn_regression(data_set, y_values, feature_types):
    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=3600,
        per_run_time_limit=300,
        tmp_folder='/tmp/autosklearn_regression_example_tmp',
        output_folder='/tmp/autosklearn_regression_example_out',
    )
    automl.fit(data_set, y_values,
               dataset_name='boston',
               feat_type=feature_types)
'''
