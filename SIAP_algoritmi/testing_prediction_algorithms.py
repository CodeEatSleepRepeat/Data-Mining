import dimension_reduction_algorithms
import prediction_algorithms
import validation_methods
import utils
import pandas as pd

'''
COLLECTING DATA SET FROM URL AND SPLITTING TO TRAIN/TEST SETS
'''
collected_data = utils.read_exel("/home/sale/Desktop/GitHubSIAP/Data-Mining/SIAP_algoritmi/human-rights-scores.xlsx")
data_set = collected_data[['Country Code', 'Year', 'Human Rights Protection Scores']]
# x_train, x_test, y_train, y_test = utils.train_test_split_data(data_set[:, 0:2], data_set[:, 2], 0.2)
x_train, x_test, y_train, y_test = utils.train_test_split_data(data_set[['Country Code', 'Year']],
                                                               data_set[['Human Rights Protection Scores']], 0.2)

'''
ONE HOT ENCODING CALLING
'''
# kako bi uradili one hot encodovanje preko pandasa dovoljno je samo ovo da pozovemo i on ce sve kategorije da prebaci kao nove kolone dataseta...
x_train_one_hot = pd.get_dummies(collected_data)
# df_dummy = pd.get_dummies(df_r, drop_first=True) mozda ovo drop first odbacuje indekse pa je dobro dodati
# print(x_train_one_hot.head())


'''
LABEL ENCODING CALLING
'''
# za label encoder treba da sse prosledi dataFrame objekat i koje kolone zelimo da izmenimo
x_train_label_encoder = utils.label_encoding2(x_train, 'Country Code')
x_test_label_encoder = utils.label_encoding2(x_test, 'Country Code')
x_train_label_encoder.sort_index(inplace=True)
# print(x_train_label_encoder.to_string())


'''
CALLING X BOOST REGRESSION AND PRINTING ITS RESULT WITH MSE AND R2 ERROR 
'''
x_Boost_regression = prediction_algorithms.XGBoost_regression(x_train_label_encoder, y_train)
xBoost_prediction = x_Boost_regression.predict(
    x_test_label_encoder)  # za predikciju neophono je da prosledimo 2d array, odnosno ceo dataset wtf

MSE = validation_methods.mean_squared_error_metrics(y_test, xBoost_prediction)
r2_error = validation_methods.r2_metrics(y_test, xBoost_prediction)
print("X boost regression errors:")
print("MSE: ")
print(MSE)
print("R2 ERROR: ")
print(r2_error)

'''
CALLING COX REGRESSION AND PRINTING ITS RESULT WITH MSE AND R2 ERROR

Beside encoded data_set cox regression needs to know which column represents duration column and which column needs
to be predicted 'event column'

ne znam kako ovo da testiram
'''
data_set_encoded = utils.label_encoding2(data_set, 'Country Code')
cox_regression = prediction_algorithms.cox_regression(data_set_encoded, "Year", "Human Rights Protection Scores")
# cox_regression.print_summary()

'''
ELASTIC NET REGRESSION

'''
elastic_regression = prediction_algorithms.elastic_regression(x_train_label_encoder, y_train, 0.01)
elastic_regression_prediction = elastic_regression.predict(x_test_label_encoder)

MSE = validation_methods.mean_squared_error_metrics(y_test, elastic_regression_prediction)
r2_error = validation_methods.r2_metrics(y_test, elastic_regression_prediction)
print("Elastic net regression errors:")
print("MSE: ")
print(MSE)
print("R2 ERROR: ")
print(r2_error)

'''
PARTIAL LEAST SQUARES PLS

On odma stampa rezultate, i koliko vidim ima priblizne kao predhodna 2 algoritma koja sam uspeo istestirati
'''
prediction_algorithms.partial_least_squares(x_train_label_encoder, y_train, 3, True)
