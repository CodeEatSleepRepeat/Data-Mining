import utils
import prediction_algorithms
import pandas as pd
import validation_methods
import math
import numpy as np


# saving csv files into forwarded path
def save_data_as_csv(data, path):
    import csv
    import os

    filename = os.path.join(path + ".csv")
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data)


def print_validation_scores(algorithm_prediction, true_value):
    # testiranje na gresku
    MSE = validation_methods.mean_squared_error_metrics(true_value, algorithm_prediction)
    r2_error = validation_methods.r2_metrics(true_value, algorithm_prediction)
    print("X boost regression errors:")
    print("MSE: ")
    print(MSE)
    print("R2 ERROR: ")
    print(r2_error)


def predict_data_set(x_train, x_test, y_train, y_test, data_set_name, fill_with, algorithm):
    print('############################################################################################')
    print('############################################################################################')

    print(
        'PREDICTION RESULTS OF: ' + data_set_name + " WITH FILL METHOD: " + fill_with + " AND " + algorithm + " ALGORITHM")

    if fill_with == "mean":
        fill_with_value_x_train = x_train.mean()
        fill_with_value_x_test = x_test.mean()
        fill_with_value_y_train = y_train.mean()
        fill_with_value_y_test = y_test.mean()
    elif fill_with == "min":
        fill_with_value_x_train = x_train.min()
        fill_with_value_x_test = x_test.min()
        fill_with_value_y_train = y_train.min()
        fill_with_value_y_test = y_test.min()
    elif fill_with == "max":
        fill_with_value_x_train = x_train.max()
        fill_with_value_x_test = x_test.max()
        fill_with_value_y_train = y_train.max()
        fill_with_value_y_test = y_test.max()
    else:
        fill_with_value_x_train = 0
        fill_with_value_x_test = 0
        fill_with_value_y_train = 0
        fill_with_value_y_test = 0

    # popunjavamo NaN vrednosti sa srednjom vrednosti dataseta jer algoritam ne dozvoljava NaN vrednosti
    x_train_without_nan = x_train.fillna(fill_with_value_x_train)
    x_test_without_nan = x_test.fillna(fill_with_value_x_test)
    y_train_without_nan = y_train.fillna(fill_with_value_y_train)
    y_test_without_nan = y_test.fillna(fill_with_value_y_test)

    if algorithm == "xboost":
        # treniranje predikcionog modela x boost regression
        x_Boost_regression = prediction_algorithms.XGBoost_regression(pd.get_dummies(x_train_without_nan),
                                                                      y_train_without_nan)
        xBoost_prediction = x_Boost_regression.predict(
            x_test_without_nan)  # za predikciju neophono je da prosledimo 2d array, odnosno ceo dataset wtf
        print_validation_scores(xBoost_prediction, y_test_without_nan)

        return x_Boost_regression
    elif algorithm == "elastic":
        elastic_regression = prediction_algorithms.elastic_regression(x_train_without_nan,
                                                                      y_train_without_nan, 0.01)
        elastic_regression_prediction = elastic_regression.predict(x_test_without_nan)

        print("DIMENZIJA OD MODELA JE: " + str(x_test_without_nan.shape))
        print_validation_scores(elastic_regression_prediction, y_test_without_nan)

        return elastic_regression


'''
    x_value_encoded  => enkoridani x podaci, potrebni za regresioni model.
    x_value          => originalni podaci, gde je ime drzave predstavljeno preko ISO-3 standarda u jednoj koloni
                        , korisceno za formiranje dokumenta (da se ne stavlja enkodirana vrednost, jer nije potrebno)
    y_value          => sadrzi vrednosti koje hocemo da popunimo, ukoliko je NaN jedna od vrednosti, uz pomoc modela je punimo
                        , ukoliko ima vrednost, ostavljamo je
    prediction_model => regresioni model koji se koristi za predvidjanje y vrednosti koja ima NaN vrednost
'''


def fill_data_sets(prediction_model, x_value_encoded, x_value, y_value, name_of_data_set):
    new_data_set = []

    number_of_nans = 0
    for i in range(0, len(y_value)):

        x_value_row = x_value.iloc[i:i + 1, :]
        x_value_row_value = x_value_row.values[0]
        x_value_encoded_row = x_value_encoded.iloc[i:i + 1, :]

        if math.isnan(y_value.iloc[i]):
            predicted_value = prediction_model.predict(x_value_encoded_row)
            x_value_row_value = np.append(x_value_row_value, predicted_value)
            new_data_set.append(x_value_row_value)
            number_of_nans = number_of_nans + 1
        else:
            x_value_row_value = np.append(x_value_row_value, y_value.iloc[i].values[0])
            new_data_set.append(x_value_row_value)

    print("Number of NaNs: " + str(number_of_nans))

    print("Prosao, sad cuvanje")
    save_data_as_csv(new_data_set,
                     "/home/sale/PycharmProjects/fillovane_tabele/" + name_of_data_set)
    print("sacuvano")


'''
    MAIN
    DATA SETOVI
'''

'''
AGRICULTURAL METHANE EMISSION DATA SET
'''

agricultural_methane_emission = utils.read_exel(
    "/home/sale/Desktop/GitHubSIAP/Data-Mining/tabele1990-2017/treba_regresija/agricultural_methane_emission.xlsx")
# cist ispis da proverimo da li je tabela dobro ucitana.
# print(agricultural_methane_emission.head())
# enkodujemo x_data... sa one hot encodingom, jer sadrzi kolonu year koja je string kategorickog tipa
x_data_agricultural_methane_emission = agricultural_methane_emission[['Country Code', 'Year']]
y_data_agricultural_methane_emission = agricultural_methane_emission[['Agricultural methane emissions (% of total)']]

x_data_encoded_agricultural_methane_emission = pd.get_dummies(x_data_agricultural_methane_emission)

x_train_agricultural_methane_emission, x_test_agricultural_methane_emission, y_train_agricultural_methane_emission, y_test_agricultural_methane_emission = utils.train_test_split_data(
    x_data_encoded_agricultural_methane_emission,
    y_data_agricultural_methane_emission,
    0.2)

prediction_model_agrocultural = predict_data_set(x_train_agricultural_methane_emission,
                                                 x_test_agricultural_methane_emission,
                                                 y_train_agricultural_methane_emission,
                                                 y_test_agricultural_methane_emission,
                                                 'AGRICULTURAL METHANE EMISSION DATA SET', 'mean', 'elastic')

fill_data_sets(prediction_model_agrocultural, x_data_encoded_agricultural_methane_emission,
               x_data_agricultural_methane_emission,
               y_data_agricultural_methane_emission, "agricultural_methane_emission_filled_data_set")

'''
CORRUPTION PERCEPTIONS INDEX DATA SET
'''
corruption_perceptions_index = utils.read_exel(
    "/home/sale/Desktop/GitHubSIAP/Data-Mining/tabele1990-2017/treba_regresija/corruption_perceptions_index.xlsx")
# cist ispis da proverimo da li je tabela dobro ucitana.
# print(corruption_perceptions_index.head())
# enkodujemo x_data... sa one hot encodingom, jer sadrzi kolonu year koja je string kategorickog tipa
x_data_corruption_perceptions_index = corruption_perceptions_index[['Country Code', 'Year']]
y_data_corruption_perceptions_index = corruption_perceptions_index[['Corruption index']]

x_data_encoded_corruption_perceptions_index = pd.get_dummies(x_data_corruption_perceptions_index)
x_train_corruption_perceptions_index, x_test_corruption_perceptions_index, y_train_corruption_perceptions_index, y_test_corruption_perceptions_index = utils.train_test_split_data(
    x_data_encoded_corruption_perceptions_index,
    y_data_corruption_perceptions_index,
    0.2)

prediction_model_corruption_perceptions_index =predict_data_set(x_train_corruption_perceptions_index, x_test_corruption_perceptions_index,
                 y_train_corruption_perceptions_index, y_test_corruption_perceptions_index,
                'CORRUPTION PERCEPTIONS INDEX DATA SET', 'min', 'elastic')

fill_data_sets(prediction_model_corruption_perceptions_index, x_data_encoded_corruption_perceptions_index,
               x_data_corruption_perceptions_index,
               y_data_corruption_perceptions_index, "corruption_perceptions_index_filled_data_set")

'''
CORRUPTION PERCEPTION INDEX 2 DATASET
'''

coruption_perception_index_2 = utils.read_exel(
    "/home/sale/Desktop/GitHubSIAP/Data-Mining/tabele1990-2017/treba_regresija/CoruptionPerceptionIndex_filtered.xlsx")
# cist ispis da proverimo da li je tabela dobro ucitana.
# print(coruption_perception_index_2.head())
# enkodujemo x_data... sa one hot encodingom, jer sadrzi kolonu year koja je string kategorickog tipa
x_data_coruption_perception_index_2 = coruption_perception_index_2[['Country Code', 'Year']]
y_data_coruption_perception_index_2 = coruption_perception_index_2[['corruption_index']]

x_data_encoded_coruption_perception_index_2 = pd.get_dummies(x_data_coruption_perception_index_2)
x_train_coruption_perception_index_2, x_test_coruption_perception_index_2, y_train_coruption_perception_index_2, y_test_coruption_perception_index_2 = utils.train_test_split_data(
    x_data_encoded_coruption_perception_index_2,
    y_data_coruption_perception_index_2,
    0.2)

prediction_model_coruption_perception_index_2 = predict_data_set(x_train_coruption_perception_index_2, x_test_coruption_perception_index_2,
                y_train_coruption_perception_index_2, y_test_coruption_perception_index_2,
                'CORRUPTION PERCEPTION INDEX 2 DATASET', 'mean', 'elastic')

fill_data_sets(prediction_model_coruption_perception_index_2, x_data_encoded_coruption_perception_index_2,
               x_data_coruption_perception_index_2,
               y_data_coruption_perception_index_2, "coruption_perception_index_2_filled_data_set")

'''
DAILY PER CAPITA SUPPLY OF CALORIES DATA SET
'''

daily_per_capita_supply_of_calories = utils.read_exel(
    "/home/sale/Desktop/GitHubSIAP/Data-Mining/tabele1990-2017/treba_regresija/daily-per-capita-supply-of-calories.xlsx")
# cist ispis da proverimo da li je tabela dobro ucitana.
# print(daily_per_capita_supply_of_calories.head())
# enkodujemo x_data... sa one hot encodingom, jer sadrzi kolonu year koja je string kategorickog tipa
x_data_daily_per_capita_supply_of_calories = daily_per_capita_supply_of_calories[['Country Code', 'Year']]
y_data_daily_per_capita_supply_of_calories = daily_per_capita_supply_of_calories[['Daily caloric supply (kcal/person/day)']]

x_data_encoded_daily_per_capita_supply_of_calories = pd.get_dummies(x_data_daily_per_capita_supply_of_calories)
x_train_daily_per_capita_supply_of_calories, x_test_daily_per_capita_supply_of_calories, y_train_daily_per_capita_supply_of_calories, y_test_daily_per_capita_supply_of_calories = utils.train_test_split_data(
    x_data_encoded_daily_per_capita_supply_of_calories,
    y_data_daily_per_capita_supply_of_calories,
    0.2)

prediction_model_daily_per_capita_supply_of_calories = predict_data_set(x_train_daily_per_capita_supply_of_calories, x_test_daily_per_capita_supply_of_calories,
                y_train_daily_per_capita_supply_of_calories, y_test_daily_per_capita_supply_of_calories,
                'DAILY PER CAPITA SUPPLY OF CALORIES DATA SET', 'mean', 'elastic')

fill_data_sets(prediction_model_daily_per_capita_supply_of_calories, x_data_encoded_daily_per_capita_supply_of_calories,
               x_data_daily_per_capita_supply_of_calories,
               y_data_daily_per_capita_supply_of_calories, "daily_per_capita_supply_of_calories_filled_data_set")

'''
LIFE SATISFACTION DATA SET
'''
life_satisfaction = utils.read_exel(
    "/home/sale/Desktop/GitHubSIAP/Data-Mining/tabele1990-2017/treba_regresija/life_satisfaction.xlsx")
# cist ispis da proverimo da li je tabela dobro ucitana.
# print(life_satisfaction.head())
# enkodujemo x_data... sa one hot encodingom, jer sadrzi kolonu year koja je string kategorickog tipa
x_data_life_satisfaction = life_satisfaction[['Country Code', 'Year']]
y_data_life_satisfaction = life_satisfaction[['Happines']]

x_data_encoded_life_satisfaction = pd.get_dummies(x_data_life_satisfaction)
x_train_life_satisfaction, x_test_life_satisfaction, y_train_life_satisfaction, y_test_life_satisfaction = utils.train_test_split_data(
    x_data_encoded_life_satisfaction,
    y_data_life_satisfaction,
    0.2)

prediction_model_life_satisfaction = predict_data_set(x_train_life_satisfaction, x_test_life_satisfaction, y_train_life_satisfaction,
                y_test_life_satisfaction, 'LIFE SATISFACTION DATA SET', 'mean', 'elastic')


fill_data_sets(prediction_model_life_satisfaction, x_data_encoded_life_satisfaction,
               x_data_life_satisfaction,
               y_data_life_satisfaction, "life_satisfaction_filled_data_set")

'''
POLITICAL REGIME UPDATED2016 DATA SET
'''
political_regime_updated2016 = utils.read_exel(
    "/home/sale/Desktop/GitHubSIAP/Data-Mining/tabele1990-2017/treba_regresija/political-regime-updated2016.xlsx")
# cist ispis da proverimo da li je tabela dobro ucitana.
# print(political_regime_updated2016.head())
# enkodujemo x_data... sa one hot encodingom, jer sadrzi kolonu year koja je string kategorickog tipa
x_data_political_regime_updated2016 = political_regime_updated2016[['Country Code', 'Year']]
y_data_political_regime_updated2016 = political_regime_updated2016[['Political Regime (OWID based on Polity IV and Wimmer & Min) (Score)']]

x_data_encoded_political_regime_updated2016 = pd.get_dummies(x_data_political_regime_updated2016)
x_train_political_regime_updated2016, x_test_political_regime_updated2016, y_train_political_regime_updated2016, y_test_political_regime_updated2016 = utils.train_test_split_data(
    x_data_encoded_political_regime_updated2016,
    y_data_political_regime_updated2016,
    0.2)

prediction_model_political_regime_updated2016 = predict_data_set(x_train_political_regime_updated2016, x_test_political_regime_updated2016,
                y_train_political_regime_updated2016, y_test_political_regime_updated2016,
                'POLITICAL REGIME UPDATED2016 DATA SET', 'mean', 'elastic')


fill_data_sets(prediction_model_political_regime_updated2016, x_data_encoded_political_regime_updated2016,
               x_data_political_regime_updated2016,
               y_data_political_regime_updated2016, "political_regime_updated2016_filled_data_set")

'''
SHARE WITH ALCOHOL OR DRUG USE DISORDERS 
'''
share_with_alcohol_or_drug_use_disorders = utils.read_exel(
    "/home/sale/Desktop/GitHubSIAP/Data-Mining/tabele1990-2017/treba_regresija/share-with-alcohol-or-drug-use-disorders.xlsx")
# cist ispis da proverimo da li je tabela dobro ucitana.
# print(share_with_alcohol_or_drug_use_disorders.head())
# enkodujemo x_data... sa one hot encodingom, jer sadrzi kolonu year koja je string kategorickog tipa
x_data_share_with_alcohol_or_drug_use_disorders = share_with_alcohol_or_drug_use_disorders[['Country Code', 'Year']]
y_data_share_with_alcohol_or_drug_use_disorders = share_with_alcohol_or_drug_use_disorders[['Prevalence - Alcohol and substance use disorders: Both (age-standardized percent) (%)']]

x_data_encoded_share_with_alcohol_or_drug_use_disorders = pd.get_dummies(x_data_share_with_alcohol_or_drug_use_disorders)
x_train_share_with_alcohol_or_drug_use_disorders, x_test_share_with_alcohol_or_drug_use_disorders, y_train_share_with_alcohol_or_drug_use_disorders, y_test_share_with_alcohol_or_drug_use_disorders = utils.train_test_split_data(
    x_data_encoded_share_with_alcohol_or_drug_use_disorders,
    y_data_share_with_alcohol_or_drug_use_disorders,
    0.2)

prediction_model_share_with_alcohol_or_drug_use_disorders = predict_data_set(x_train_share_with_alcohol_or_drug_use_disorders, x_test_share_with_alcohol_or_drug_use_disorders,
                y_train_share_with_alcohol_or_drug_use_disorders, y_test_share_with_alcohol_or_drug_use_disorders,
                'SHARE WITH ALCOHOL OR DRUG USE DISORDERS', 'mean', 'elastic')


fill_data_sets(prediction_model_share_with_alcohol_or_drug_use_disorders, x_data_encoded_share_with_alcohol_or_drug_use_disorders,
               x_data_share_with_alcohol_or_drug_use_disorders,
               y_data_share_with_alcohol_or_drug_use_disorders, "share_with_alcohol_or_drug_use_disorders_filled_data_set")

'''
UN MIGRANT STOCK BY ORIGIN AND DESTINATION 2019
'''
UN_MigrantStockByOriginAndDestination_2019 = utils.read_exel(
    "/home/sale/Desktop/GitHubSIAP/Data-Mining/tabele1990-2017/treba_regresija/UN_MigrantStockByOriginAndDestination_2019.xlsx")
# cist ispis da proverimo da li je tabela dobro ucitana.
# print(UN_MigrantStockByOriginAndDestination_2019.head())
# enkodujemo x_data... sa one hot encodingom, jer sadrzi kolonu year koja je string kategorickog tipa
x_data_UN_MigrantStockByOriginAndDestination_2019 = UN_MigrantStockByOriginAndDestination_2019[['Country Code', 'Year']]
y_data_UN_MigrantStockByOriginAndDestination_2019 = UN_MigrantStockByOriginAndDestination_2019[['Total origin']]

x_data_encoded_UN_MigrantStockByOriginAndDestination_2019 = pd.get_dummies(x_data_UN_MigrantStockByOriginAndDestination_2019)
x_train_UN_MigrantStockByOriginAndDestination_2019, x_test_UN_MigrantStockByOriginAndDestination_2019, y_train_UN_MigrantStockByOriginAndDestination_2019, y_test_UN_MigrantStockByOriginAndDestination_2019 = utils.train_test_split_data(
    x_data_encoded_UN_MigrantStockByOriginAndDestination_2019,
    y_data_UN_MigrantStockByOriginAndDestination_2019,
    0.2)

prediction_model_UN_MigrantStockByOriginAndDestination_2019 = predict_data_set(x_train_UN_MigrantStockByOriginAndDestination_2019, x_test_UN_MigrantStockByOriginAndDestination_2019,
                y_train_UN_MigrantStockByOriginAndDestination_2019, y_test_UN_MigrantStockByOriginAndDestination_2019,
                'UN MIGRANT STOCK BY ORIGIN AND DESTINATION 2019', 'mean', 'elastic')

fill_data_sets(prediction_model_UN_MigrantStockByOriginAndDestination_2019, x_data_encoded_UN_MigrantStockByOriginAndDestination_2019,
               x_data_UN_MigrantStockByOriginAndDestination_2019,
               y_data_UN_MigrantStockByOriginAndDestination_2019, "UN_MigrantStockByOriginAndDestination_2019_filled_data_set")

'''
WARS FORMATTED
'''
wars_formated = utils.read_exel(
    "/home/sale/Desktop/GitHubSIAP/Data-Mining/tabele1990-2017/treba_regresija/wars_formated.xlsx")
# cist ispis da proverimo da li je tabela dobro ucitana.
# print(wars_formated.head())
# enkodujemo x_data... sa one hot encodingom, jer sadrzi kolonu year koja je string kategorickog tipa
x_data_encoded_wars_formated = pd.get_dummies(wars_formated[['Country Code', 'Year']])
x_train_wars_formated, x_test_wars_formated, y_train_wars_formated, y_test_wars_formated = utils.train_test_split_data(
    x_data_encoded_wars_formated,
    wars_formated[
        ['won_war']],
    0.2)

# predict_data_set(x_train_wars_formated, x_test_wars_formated, y_train_wars_formated, y_test_wars_formated,
#                'WARS FORMATTED DATA SET')

'''
FREEDOM OF PRESS DATA, LEGAL ENVIRONMENT AS PREDICTING VALUE
'''
freedom_of_the_press_data_legal_environment = utils.read_exel(
    "/home/sale/Desktop/GitHubSIAP/Data-Mining/tabele1990-2017/treba_regresija/Freedom_of_the_Press_Data.xlsx")
# cist ispis da proverimo da li je tabela dobro ucitana.
# print(freedom_of_the_press_data_legal_environment.head())
# enkodujemo x_data... sa one hot encodingom, jer sadrzi kolonu year koja je string kategorickog tipa
x_data_freedom_of_the_press_data_legal_environment = freedom_of_the_press_data_legal_environment[['Country Code', 'Year','Political environment','Economic environment', 'Total Score']]
y_data_freedom_of_the_press_data_legal_environment = freedom_of_the_press_data_legal_environment[['Legal environment']]

x_data_encoded_freedom_of_the_press_data_legal_environment = pd.get_dummies(x_data_freedom_of_the_press_data_legal_environment)
x_train_freedom_of_the_press_data_legal_environment, x_test_freedom_of_the_press_data_legal_environment, y_train_freedom_of_the_press_data_legal_environment, y_test_freedom_of_the_press_data_legal_environment = utils.train_test_split_data(
    x_data_encoded_freedom_of_the_press_data_legal_environment,
    y_data_freedom_of_the_press_data_legal_environment,
    0.2)

prediction_model_freedom_of_the_press_data_legal_environment = predict_data_set(x_train_freedom_of_the_press_data_legal_environment,
               x_test_freedom_of_the_press_data_legal_environment,
               y_train_freedom_of_the_press_data_legal_environment,
               y_test_freedom_of_the_press_data_legal_environment,
               'FREEDOM OF PRESS DATA, LEGAL ENVIRONMENT AS PREDICTING VALUE', 'zero', 'elastic')


fill_data_sets(prediction_model_freedom_of_the_press_data_legal_environment, x_data_encoded_freedom_of_the_press_data_legal_environment.fillna(0),
               x_data_freedom_of_the_press_data_legal_environment.fillna(0),
               y_data_freedom_of_the_press_data_legal_environment, "freedom_of_the_press_data_legal_environment_filled_data_set")

'''
FREEDOM OF PRESS DATA, POLITICAL ENVIRONMENT AS PREDICTING VALUE
'''
freedom_of_the_press_data_political_environment = utils.read_exel(
    "/home/sale/Desktop/GitHubSIAP/Data-Mining/tabele1990-2017/treba_regresija/Freedom_of_the_Press_Data.xlsx")
# cist ispis da proverimo da li je tabela dobro ucitana.
# print(freedom_of_the_press_data_political_environment.head())
# enkodujemo x_data... sa one hot encodingom, jer sadrzi kolonu year koja je string kategorickog tipa
x_data_freedom_of_the_press_data_political_environment = freedom_of_the_press_data_political_environment[['Country Code', 'Year', 'Legal environment', 'Economic environment', 'Total Score']]
y_data_freedom_of_the_press_data_political_environment = freedom_of_the_press_data_political_environment[['Political environment']]

x_data_encoded_freedom_of_the_press_data_political_environment = pd.get_dummies(x_data_freedom_of_the_press_data_political_environment)
x_train_freedom_of_the_press_data_political_environment, x_test_freedom_of_the_press_data_political_environment, y_train_freedom_of_the_press_data_political_environment, y_test_freedom_of_the_press_data_political_environment = utils.train_test_split_data(
    x_data_encoded_freedom_of_the_press_data_political_environment,
    y_data_freedom_of_the_press_data_political_environment,
    0.2)

prediction_model_freedom_of_the_press_data_political_environment = predict_data_set(x_train_freedom_of_the_press_data_political_environment,
                x_test_freedom_of_the_press_data_political_environment,
                y_train_freedom_of_the_press_data_political_environment,
                y_test_freedom_of_the_press_data_political_environment,
                'FREEDOM OF PRESS DATA, POLITICAL ENVIRONMENT AS PREDICTING VALUE', 'zero', 'elastic')


fill_data_sets(prediction_model_freedom_of_the_press_data_political_environment, x_data_encoded_freedom_of_the_press_data_political_environment.fillna(0),
               x_data_freedom_of_the_press_data_political_environment.fillna(0),
               y_data_freedom_of_the_press_data_political_environment, "freedom_of_the_press_data_political_environment_filled_data_set")


'''
FREEDOM OF PRESS DATA, ECONOMIC ENVIRONMENT AS PREDICTING VALUE
'''
freedom_of_the_press_data_economic_environment = utils.read_exel(
    "/home/sale/Desktop/GitHubSIAP/Data-Mining/tabele1990-2017/treba_regresija/Freedom_of_the_Press_Data.xlsx")
# cist ispis da proverimo da li je tabela dobro ucitana.
# print(freedom_of_the_press_data_economic_environment.head())
# enkodujemo x_data... sa one hot encodingom, jer sadrzi kolonu year koja je string kategorickog tipa
x_data_freedom_of_the_press_data_economic_environment = freedom_of_the_press_data_economic_environment[['Country Code', 'Year', 'Legal environment', 'Political environment', 'Total Score']]
y_data_freedom_of_the_press_data_economic_environment = freedom_of_the_press_data_economic_environment[['Economic environment']]

x_data_encoded_freedom_of_the_press_data_economic_environment = pd.get_dummies(x_data_freedom_of_the_press_data_economic_environment)
x_train_freedom_of_the_press_data_economic_environment, x_test_freedom_of_the_press_data_economic_environment, y_train_freedom_of_the_press_data_economic_environment, y_test_freedom_of_the_press_data_economic_environment = utils.train_test_split_data(
    x_data_encoded_freedom_of_the_press_data_economic_environment,
    y_data_freedom_of_the_press_data_economic_environment,
    0.2)

prediction_model_freedom_of_the_press_data_economic_environment = predict_data_set(x_train_freedom_of_the_press_data_economic_environment,
               x_test_freedom_of_the_press_data_economic_environment,
               y_train_freedom_of_the_press_data_economic_environment,
               y_test_freedom_of_the_press_data_economic_environment,
              'FREEDOM OF PRESS DATA, ECONOMIC ENVIRONMENT AS PREDICTING VALUE', 'zero', 'elastic')


fill_data_sets(prediction_model_freedom_of_the_press_data_economic_environment, x_data_encoded_freedom_of_the_press_data_economic_environment.fillna(0),
               x_data_freedom_of_the_press_data_economic_environment.fillna(0),
               y_data_freedom_of_the_press_data_economic_environment, "freedom_of_the_press_data_economic_environment_filled_data_set")


'''
FREEDOM OF PRESS DATA, TOTAL SCORE AS PREDICTING VALUE
'''
freedom_of_the_press_data_total_score = utils.read_exel(
    "/home/sale/Desktop/GitHubSIAP/Data-Mining/tabele1990-2017/treba_regresija/Freedom_of_the_Press_Data.xlsx")
# cist ispis da proverimo da li je tabela dobro ucitana.
# print(freedom_of_the_press_data_total_score.head())
# enkodujemo x_data... sa one hot encodingom, jer sadrzi kolonu year koja je string kategorickog tipa
x_data_freedom_of_the_press_data_total_score = freedom_of_the_press_data_total_score[['Country Code', 'Year', 'Legal environment', 'Political environment', 'Economic environment']]
y_data_freedom_of_the_press_data_total_score = freedom_of_the_press_data_total_score[['Total Score']]

x_data_encoded_freedom_of_the_press_data_total_score = pd.get_dummies(x_data_freedom_of_the_press_data_total_score)
x_train_freedom_of_the_press_data_total_score, x_test_freedom_of_the_press_data_total_score, y_train_freedom_of_the_press_data_total_score, y_test_freedom_of_the_press_data_total_score = utils.train_test_split_data(
    x_data_encoded_freedom_of_the_press_data_total_score,
    y_data_freedom_of_the_press_data_total_score,
    0.2)

prediction_model_freedom_of_the_press_data_total_score = predict_data_set(x_train_freedom_of_the_press_data_total_score, x_test_freedom_of_the_press_data_total_score,
               y_train_freedom_of_the_press_data_total_score, y_test_freedom_of_the_press_data_total_score,
               'FREEDOM OF PRESS DATA, TOTAL SCORE AS PREDICTING VALUE', 'zero', 'elastic')


fill_data_sets(prediction_model_freedom_of_the_press_data_total_score, x_data_encoded_freedom_of_the_press_data_total_score.fillna(0),
               x_data_freedom_of_the_press_data_total_score.fillna(0),
               y_data_freedom_of_the_press_data_total_score, "freedom_of_the_press_data_total_score_filled_data_set")
