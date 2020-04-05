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

'''
    Funkcija koja ispisuje vrednosti validacionih algoritama na gresku
'''
def print_validation_scores(algorithm_prediction, true_value):
    # testiranje na gresku
    MSE = validation_methods.mean_squared_error_metrics(true_value, algorithm_prediction)
    r2_error = validation_methods.r2_metrics(true_value, algorithm_prediction)
    print("X boost regression errors:")
    print("MSE: ")
    print(MSE)
    print("R2 ERROR: ")
    print(r2_error)

'''
    Funkcija koja kao parametre prima:
        - URL putanju do dokumenta
        - Informaciju o tipu dokumenta (csv ili exel)
        - Nazive kolona koji ce se koristit za prediktore
        - Naziv kolone koju treba prediktovati
        
    Funkcija vraca:
        - Ucitani data frame (document)
        - Kolone prediktora (x_data) i predikcione kolone (y_data)
        - Enkodovane kolone prediktora (x_data_encoded)
        - I splitovan data set u 4 skupa : x_test, x_train, y_test, y_train
'''
def get_important_data(doc_url_path, doc_type, x_data_column_names, y_data_column_name):
    document = {}
    if doc_type == "csv":
        document = utils.read_csv(doc_url_path)
    elif doc_type == "exel":
        document = utils.read_exel(doc_url_path)

    x_data = document[x_data_column_names]
    y_data = document[y_data_column_name]

    x_data_encoded = pd.get_dummies(x_data)

    x_train, x_test, y_train, y_test = utils.train_test_split_data(x_data_encoded, y_data, 0.2)

    return document, x_data, y_data, x_data_encoded, x_train, x_test, y_train, y_test


'''
    Funkcija koja za rezultat vraca istreniran regresioni model
    
    Na ulaz prima:
        - Splitovan pocetni data set (x_train, x_test, y_train, y_test)
        - Naziv koji zelimo da ispisemo u konzolnom ispisu za data set
        - Naziv metoda sa kojim zelimo da upotpunimo nedostajuce vrednost (fill_with): mean, min, max, zero..
        - Naziv algoritma koji zelimo da istreniramo (algorithm)

'''

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
                     destination_to_save_new_documents + name_of_data_set)
    print("sacuvano")


'''
    Funkcija koja vraca mapu drzava = lista_nedostajucih_godina. 
    Ulaz u funkciju je sam data set i opseg godina koje zelimo da ispitamo
'''


def get_country_missing_years_map(data_set, year_from, year_to):
    all_years = list(range(year_from, year_to + 1))
    country_to_missing_years_map = {}

    for row in data_set.values:

        if row[0] in country_to_missing_years_map:
            if row[1] in country_to_missing_years_map[row[0]]:
                country_to_missing_years_map[row[0]].remove(row[1])
        else:
            country_to_missing_years_map[row[0]] = all_years

    return country_to_missing_years_map


'''
    Funkcija koja za rezultat vraca data set koji pokriva sve neophodne godine gde ce na novododatim
    godinama na y vrednosti stojati NaN
    
    Na ulazu dobija:
        - data_set koji treba dopuniti za nedostajuce godine
        - dictionary koji kao kljuc sadrzi ime drzave, a na vrednosti listu svih godina koje nisu
            zabelezeni za tu drzavu, a neophodne su
'''

def fill_missing_years_for_countries_in_data_set_with_nan(data_set, country_to_missing_years_map):
    data_set_values = []

    for data in data_set.values:
        data_set_values.append(data)

    for country in country_to_missing_years_map:
        list_of_missing_years = country_to_missing_years_map[country]
        for missing_year in list_of_missing_years:
            new_row = [country, missing_year, float('nan')]
            data_set_values.append(new_row)



    return data_set_values


'''
    Nacin na koji nadohnadjujemo nedostajuce godine i laziramo vrednosti:

    1. Pronalazimo listu svih drzava koje se nalaze u data setu i uz svaku drzavu postavljena je lista
        sa godinama koje nedostaju u datasetu (npr za nas rad je bitan raspon od 1990-2017)

    2. Nakon sto smo napravili recnik sa tim vrednostima, pravimo novi data_set koji ce sadrzati i kolone
        drzava sa nedostajucim godinama, sem sto ce na mestu "predvidjane vrednosti" stojati NaN vrednost

    3. Pretvaramo data_set u Data_frame, delimo ga na X i Y podsetove i Xencoded podset

    4. Prosledjujemo funkciji fill_data_sets sve podatke i ona puni NaN vrednosti sa predvidjenim vrednostima
        i cuva u fajl.csv
'''


def fill_missing_year_data(document, prediction_model, x_data_column_names, y_data_column_name, year_from, year_to,
                           file_name):
    # 1
    country_years_map = get_country_missing_years_map(document, year_from, year_to)

    # 2
    data_with_nans = fill_missing_years_for_countries_in_data_set_with_nan(document, country_years_map)

    # 3
    data_frame = pd.DataFrame(data_with_nans,
                              columns=x_data_column_names + y_data_column_name)
    x_data_full = data_frame[x_data_column_names]
    y_data_full = data_frame[y_data_column_name]
    x_data_encoded_full = pd.get_dummies(x_data_full)

    # 4
    fill_data_sets(prediction_model, x_data_encoded_full, x_data_full, y_data_full, file_name)


'''
    SHARED FIELDS:
'''
year_from = 1990
year_to = 2017
documents_destination = "/home/sale/Desktop/GitHubSIAP/Data-Mining/josTrebaRegresija/novo/"
destination_to_save_new_documents = "/home/sale/PycharmProjects/fillovane_tabele/"

'''
    MAIN
    DATA SETOVI
'''


'''
AGRICULTURAL METHANE EMISSION DATA SET
'''

# x_data_column_name_a_m_e = ['Country Code', 'Year']
# y_data_column_name_a_m_e = ['Agricultural methane emissions (% of total)']
#
# agricultural_methane_emission, x_data_a_m_e, y_data_a_m_e, x_data_encoded_a_m_e, \
# x_train_a_m_e, x_test_a_m_e, y_train_a_m_e, y_test_a_m_e = get_important_data(
#     documents_destination + "agricultural_methane_emission_filled_data_set.xlsx",
#     "exel", x_data_column_name_a_m_e, y_data_column_name_a_m_e)
#
# prediction_model_a_m_e = predict_data_set(x_train_a_m_e,
#                                           x_test_a_m_e,
#                                           y_train_a_m_e,
#                                           y_test_a_m_e,
#                                           'AGRICULTURAL METHANE EMISSION DATA SET', 'mean',
#                                           'elastic')
#
# fill_missing_year_data(agricultural_methane_emission, prediction_model_a_m_e, x_data_column_name_a_m_e,
#                        y_data_column_name_a_m_e, year_from, year_to,
#                        'agricultural_methane_emission_filled_years_data_set')

'''
CORRUPTION PERCEPTION INDEX 2 DATASET
'''

# x_data_column_name_c_p_i_2 = ['Country Code', 'Year']
# y_data_column_name_c_p_i_2 = ['corruption_index']
#
# corruption_perception_index_2, x_data_c_p_i_2, y_data_c_p_i_2, x_data_encoded_c_p_i_2, \
# x_train_c_p_i_2, x_test_c_p_i_2, y_train_c_p_i_2, y_test_c_p_i_2 = get_important_data(
#     documents_destination + "coruption_perception_index_2_filled_data_set.xlsx",
#     "exel", x_data_column_name_c_p_i_2, y_data_column_name_c_p_i_2)
#
# prediction_model_coruption_perception_index_2 = predict_data_set(x_train_c_p_i_2,
#                                                                  x_test_c_p_i_2,
#                                                                  y_train_c_p_i_2,
#                                                                  y_test_c_p_i_2,
#                                                                  'CORRUPTION PERCEPTION INDEX 2 DATASET', 'mean',
#                                                                  'elastic')
#
# fill_missing_year_data(corruption_perception_index_2, prediction_model_coruption_perception_index_2,
#                        x_data_column_name_c_p_i_2,
#                        y_data_column_name_c_p_i_2, year_from, year_to, 'coruption_perception_index_2_filled_years')

'''
DAILY PER CAPITA SUPPLY OF CALORIES DATA SET
'''

# x_data_column_name_d_p_c_s_o_c = ['Country Code', 'Year']
# y_data_column_name_d_p_c_s_o_c = ['Daily caloric supply (kcal/person/day)']
#
# daily_per_capita_supply_of_calories, x_data_d_p_c_s_o_c, y_data_d_p_c_s_o_c, x_data_encoded_d_p_c_s_o_c, \
# x_train_d_p_c_s_o_c, x_test_d_p_c_s_o_c, y_train_d_p_c_s_o_c, y_test_d_p_c_s_o_c = get_important_data(
#     documents_destination + "daily-per-capita-supply-of-calories.xlsx",
#     "exel", x_data_column_name_d_p_c_s_o_c, y_data_column_name_d_p_c_s_o_c)
#
# prediction_model_d_p_c_s_o_c = predict_data_set(x_train_d_p_c_s_o_c,
#                                                 x_test_d_p_c_s_o_c,
#                                                 y_train_d_p_c_s_o_c,
#                                                 y_test_d_p_c_s_o_c,
#                                                 'DAILY PER CAPITA SUPPLY OF CALORIES DATA SET', 'mean',
#                                                 'elastic')
#
# fill_missing_year_data(daily_per_capita_supply_of_calories, prediction_model_d_p_c_s_o_c,
#                        x_data_column_name_d_p_c_s_o_c,
#                        y_data_column_name_d_p_c_s_o_c, year_from, year_to,
#                        'daily_per_capita_supply_of_calories_filled_years_data_set')

'''
LIFE SATISFACTION DATA SET
'''

# x_data_column_name_l_s = ['Country Code', 'Year']
# y_data_column_name_l_s = ['Happines']
#
# life_satisfaction, x_data_l_s, y_data_l_s, x_data_encoded_l_s, \
# x_train_l_s, x_test_l_s, y_train_l_s, y_test_l_s = get_important_data(
#     documents_destination + "life_satisfaction.xlsx",
#     "exel", x_data_column_name_l_s, y_data_column_name_l_s)
#
# prediction_model_l_s = predict_data_set(x_train_l_s,
#                                         x_test_l_s,
#                                         y_train_l_s,
#                                         y_test_l_s,
#                                         'LIFE SATISFACTION DATA SET', 'mean',
#                                         'elastic')
#
# fill_missing_year_data(life_satisfaction, prediction_model_l_s,
#                        x_data_column_name_l_s,
#                        y_data_column_name_l_s, year_from, year_to,
#                        'life_satisfaction_filled_years_data_set')

'''
POLITICAL REGIME UPDATED2016 DATA SET
'''

# x_data_column_name_p_r_u = ['Country Code', 'Year']
# y_data_column_name_p_r_u = ['Political Regime (OWID based on Polity IV and Wimmer & Min) (Score)']
#
# political_regime_updated2016, x_data_p_r_u, y_data_p_r_u, x_data_encoded_p_r_u, \
# x_train_p_r_u, x_test_p_r_u, y_train_p_r_u, y_test_p_r_u = get_important_data(
#     documents_destination + "political-regime-updated2016.xlsx",
#     "exel", x_data_column_name_p_r_u, y_data_column_name_p_r_u)
#
# prediction_model_p_r_u = predict_data_set(x_train_p_r_u,
#                                           x_test_p_r_u,
#                                           y_train_p_r_u,
#                                           y_test_p_r_u,
#                                           'POLITICAL REGIME UPDATED2016 DATA SET', 'mean',
#                                           'elastic')
#
# fill_missing_year_data(political_regime_updated2016, prediction_model_p_r_u,
#                        x_data_column_name_p_r_u,
#                        y_data_column_name_p_r_u, year_from, year_to,
#                        'political_regime_updated2016_filled_years_data_set')

'''
SHARE WITH ALCOHOL OR DRUG USE DISORDERS
'''
# x_data_column_name_s_w_a_o_d_u_d = ['Country Code', 'Year']
# y_data_column_name_s_w_a_o_d_u_d = [
#     'Prevalence - Alcohol and substance use disorders: Both (age-standardized percent) (%)']
#
# share_with_alcohol_or_drug_use_disorders, x_data_s_w_a_o_d_u_d, y_data_s_w_a_o_d_u_d, x_data_encoded_s_w_a_o_d_u_d, \
# x_train_s_w_a_o_d_u_d, x_test_s_w_a_o_d_u_d, y_train_s_w_a_o_d_u_d, y_test_s_w_a_o_d_u_d = get_important_data(
#     documents_destination + "share-with-alcohol-or-drug-use-disorders.xlsx",
#     "exel", x_data_column_name_s_w_a_o_d_u_d, y_data_column_name_s_w_a_o_d_u_d)
#
# prediction_model_s_w_a_o_d_u_d = predict_data_set(x_train_s_w_a_o_d_u_d,
#                                                   x_test_s_w_a_o_d_u_d,
#                                                   y_train_s_w_a_o_d_u_d,
#                                                   y_test_s_w_a_o_d_u_d,
#                                                   'SHARE WITH ALCOHOL OR DRUG USE DISORDERS', 'mean',
#                                                   'elastic')
#
# fill_missing_year_data(share_with_alcohol_or_drug_use_disorders, prediction_model_s_w_a_o_d_u_d,
#                        x_data_column_name_s_w_a_o_d_u_d,
#                        y_data_column_name_s_w_a_o_d_u_d, year_from, year_to,
#                        'share_with_alcohol_or_drug_use_disorders_filled_years_data_set')

'''
UN MIGRANT STOCK BY ORIGIN AND DESTINATION 2019
'''

# x_data_column_name_un_m_s_b_o_a_d = ['Country Code', 'Year']
# y_data_column_name_un_m_s_b_o_a_d = ['Total origin']
#
# UN_MigrantStockByOriginAndDestination_2019, x_data_un_m_s_b_o_a_d, y_data_un_m_s_b_o_a_d, x_data_encoded_un_m_s_b_o_a_d, \
# x_train_un_m_s_b_o_a_d, x_test_un_m_s_b_o_a_d, y_train_un_m_s_b_o_a_d, y_test_un_m_s_b_o_a_d = get_important_data(
#     documents_destination + "UN_MigrantStockByOriginAndDestination_2019.xlsx",
#     "exel", x_data_column_name_un_m_s_b_o_a_d, y_data_column_name_un_m_s_b_o_a_d)
#
# prediction_model_un_m_s_b_o_a_d = predict_data_set(x_train_un_m_s_b_o_a_d,
#                                                    x_test_un_m_s_b_o_a_d,
#                                                    y_train_un_m_s_b_o_a_d,
#                                                    y_test_un_m_s_b_o_a_d,
#                                                    'UN MIGRANT STOCK BY ORIGIN AND DESTINATION 2019', 'mean',
#                                                    'elastic')
#
# fill_missing_year_data(UN_MigrantStockByOriginAndDestination_2019, prediction_model_un_m_s_b_o_a_d,
#                        x_data_column_name_un_m_s_b_o_a_d,
#                        y_data_column_name_un_m_s_b_o_a_d, year_from, year_to,
#                        'UN_MigrantStockByOriginAndDestination_2019_filled_year_data_set')

'''
FREEDOM OF PRESS DATA, LEGAL ENVIRONMENT AS PREDICTING VALUE
'''
# 'Political environment', 'Economic environment', 'Total Score'
# x_data_column_name_f_o_p_d = ['Country Code', 'Year']
# y_data_column_name_f_o_p_d = ['Total Score']
#
# freedom_of_the_press_data, x_data_f_o_p_d, y_data_f_o_p_d, x_data_encoded_f_o_p_d, \
# x_train_f_o_p_d, x_test_f_o_p_d, y_train_f_o_p_d, y_test_f_o_p_d = get_important_data(
#     documents_destination + "Freedom_of_the_Press_Data.xlsx",
#     "exel", x_data_column_name_f_o_p_d, y_data_column_name_f_o_p_d)
#
# freedom_of_the_press_data = freedom_of_the_press_data[['Country Code', 'Year', 'Total Score']]
# prediction_model_f_o_p_d = predict_data_set(x_train_f_o_p_d,
#                                                    x_test_f_o_p_d,
#                                                    y_train_f_o_p_d,
#                                                    y_test_f_o_p_d,
#                                                    'FREEDOM OF PRESS DATA DATA SET', 'zero',
#                                                    'elastic')
#
# fill_missing_year_data(freedom_of_the_press_data, prediction_model_f_o_p_d,
#                        x_data_column_name_f_o_p_d,
#                        y_data_column_name_f_o_p_d, year_from, year_to,
#                        'freedom_of_press_data_filled_total_score_year_data_set')

'''
    EMPLOYERS PERCENTAGE
'''

x_data_column_name_e_p = ['Country Code', 'Year']
y_data_column_name_e_p = ['Employers, total (% of total employment) (modeled ILO estimate)']

employers_percentage, x_data_e_p, y_data_e_p, x_data_encoded_e_p, \
x_train_e_p, x_test_e_p, y_train_e_p, y_test_e_p = get_important_data(
    documents_destination + "employers_percentage.xlsx",
    "exel", x_data_column_name_e_p, y_data_column_name_e_p)

employers_percentage = employers_percentage[['Country Code', 'Year', 'Employers, total (% of total employment) (modeled ILO estimate)']]
prediction_model_e_p = predict_data_set(x_train_e_p,
                                                   x_test_e_p,
                                                   y_train_e_p,
                                                   y_test_e_p,
                                                   'EMPLOYERS PERCENTAGE', 'mean',
                                                   'elastic')

fill_missing_year_data(employers_percentage, prediction_model_e_p,
                       x_data_column_name_e_p,
                       y_data_column_name_e_p, year_from, year_to,
                       'employers_percentage_filled_year_data_set')

'''
    inflation data set
'''

x_data_column_name_i_d = ['Country Code', 'Year']
y_data_column_name_i_d = ['inflation']

inflation_data_set, x_data_i_d, y_data_i_d, x_data_encoded_i_d, \
x_train_i_d, x_test_i_d, y_train_i_d, y_test_i_d = get_important_data(
    documents_destination + "inflation.xlsx",
    "exel", x_data_column_name_i_d, y_data_column_name_i_d)

inflation_data_set = inflation_data_set[['Country Code', 'Year', 'inflation']]
prediction_model_i_d = predict_data_set(x_train_i_d,
                                                   x_test_i_d,
                                                   y_train_i_d,
                                                   y_test_i_d,
                                                   'INFLATION DATA SET', 'mean',
                                                   'elastic')

fill_missing_year_data(inflation_data_set, prediction_model_i_d,
                       x_data_column_name_i_d,
                       y_data_column_name_i_d, year_from, year_to,
                       'inflation_data_set_filled_year_data_set')

'''
   Population density data set 
'''

x_data_column_name_p_d = ['Country Code', 'Year']
y_data_column_name_p_d = ['Population density (people per sq. km of land area)']

population_density, x_data_p_d, y_data_p_d, x_data_encoded_p_d, \
x_train_p_d, x_test_p_d, y_train_p_d, y_test_p_d = get_important_data(
    documents_destination + "population_density.xlsx",
    "exel", x_data_column_name_p_d, y_data_column_name_p_d)

population_density = population_density[['Country Code', 'Year', 'Population density (people per sq. km of land area)']]
prediction_model_p_d = predict_data_set(x_train_p_d,
                                                   x_test_p_d,
                                                   y_train_p_d,
                                                   y_test_p_d,
                                                   'POPULATION DENSITY DATA SET', 'mean',
                                                   'elastic')

fill_missing_year_data(population_density, prediction_model_p_d,
                       x_data_column_name_p_d,
                       y_data_column_name_p_d, year_from, year_to,
                       'population_density_filled_year_data_set')

'''
    PTS-2019 DATA SET
'''

x_data_column_name_pts = ['Country Code', 'Year']
y_data_column_name_pts = ['PTS']

pts_2019, x_data_pts, y_data_pts, x_data_encoded_pts, \
x_train_pts, x_test_pts, y_train_pts, y_test_pts = get_important_data(
    documents_destination + "PTS-2019x.xlsx",
    "exel", x_data_column_name_pts, y_data_column_name_pts)

pts_2019 = pts_2019[['Country Code', 'Year', 'PTS']]
prediction_model_pts = predict_data_set(x_train_pts,
                                                   x_test_pts,
                                                   y_train_pts,
                                                   y_test_pts,
                                                   'PTS-2019 DATA SET', 'mean',
                                                   'elastic')

fill_missing_year_data(pts_2019, prediction_model_pts,
                       x_data_column_name_pts,
                       y_data_column_name_pts, year_from, year_to,
                       'pts_2019_filled_year_data_set')

'''
    UNEPLOYMENT DATA
'''

x_data_column_name_u_d = ['Country Code', 'Year']
y_data_column_name_u_d = ['Unemployment, total (% of total labor force) (national estimate)']

uneployment_data, x_data_u_d, y_data_u_d, x_data_encoded_u_d, \
x_train_u_d, x_test_u_d, y_train_u_d, y_test_u_d = get_important_data(
    documents_destination + "unemployment.xlsx",
    "exel", x_data_column_name_u_d, y_data_column_name_u_d)

uneployment_data = uneployment_data[['Country Code', 'Year', 'Unemployment, total (% of total labor force) (national estimate)']]
prediction_model_u_d = predict_data_set(x_train_u_d,
                                                   x_test_u_d,
                                                   y_train_u_d,
                                                   y_test_u_d,
                                                   'UNEPLOYMENT DATA SET', 'mean',
                                                   'elastic')

fill_missing_year_data(uneployment_data, prediction_model_u_d,
                       x_data_column_name_u_d,
                       y_data_column_name_u_d, year_from, year_to,
                       'uneployment_filled_year_data_set')

'''
    URBAN POPULATION GROWTH 
'''

x_data_column_name_u_p_g = ['Country Code', 'Year']
y_data_column_name_u_p_g = ['Urban population growth(%)']

urban_population_growth, x_data_u_p_g, y_data_u_p_g, x_data_encoded_u_p_g, \
x_train_u_p_g, x_test_u_p_g, y_train_u_p_g, y_test_u_p_g = get_important_data(
    documents_destination + "urban_population_growth_sredjeno.xlsx",
    "exel", x_data_column_name_u_p_g, y_data_column_name_u_p_g)

urban_population_growth = urban_population_growth[['Country Code', 'Year', 'Urban population growth(%)']]
prediction_model_u_p_g = predict_data_set(x_train_u_p_g,
                                                   x_test_u_p_g,
                                                   y_train_u_p_g,
                                                   y_test_u_p_g,
                                                   'URBAN POPULATION DATA SET', 'mean',
                                                   'elastic')

fill_missing_year_data(urban_population_growth, prediction_model_u_p_g,
                       x_data_column_name_u_p_g,
                       y_data_column_name_u_p_g, year_from, year_to,
                       'urban_population_growth_filled_year_data_set')


'''
    SCHOOLING DATA SET
'''

x_data_column_name_schooling = ['Country Code', 'Year']
y_data_column_name_schooling = ['School enrollment, tertiary (% gross)']

schooling, x_data_schooling, y_data_schooling, x_data_encoded_schooling, \
x_train_schooling, x_test_schooling, y_train_schooling, y_test_schooling = get_important_data(
    documents_destination + "schooling.xlsx",
    "exel", x_data_column_name_schooling, y_data_column_name_schooling)

schooling = schooling[['Country Code', 'Year', 'School enrollment, tertiary (% gross)']]
prediction_model_schooling = predict_data_set(x_train_schooling,
                                                   x_test_schooling,
                                                   y_train_schooling,
                                                   y_test_schooling,
                                                   'SCHOOLING DATA SET', 'mean',
                                                   'elastic')

fill_missing_year_data(schooling, prediction_model_schooling,
                       x_data_column_name_schooling,
                       y_data_column_name_schooling, year_from, year_to,
                       'schooling_1_filled_year_data_set')

'''
    SCHOOLING 2  DATA SET
'''

x_data_column_name_schooling_2 = ['Country Code', 'Year']
y_data_column_name_schooling_2 = ['School enrollment, secondary (% gross)']

schooling_2, x_data_schooling_2, y_data_schooling_2, x_data_encoded_schooling_2, \
x_train_schooling_2, x_test_schooling_2, y_train_schooling_2, y_test_schooling_2 = get_important_data(
    documents_destination + "schooling.xlsx",
    "exel", x_data_column_name_schooling_2, y_data_column_name_schooling_2)

schooling_2 = schooling_2[['Country Code', 'Year', 'School enrollment, secondary (% gross)']]
prediction_model_schooling_2 = predict_data_set(x_train_schooling_2,
                                                   x_test_schooling_2,
                                                   y_train_schooling_2,
                                                   y_test_schooling_2,
                                                   'SCHOOLING_2 DATA SET', 'mean',
                                                   'elastic')

fill_missing_year_data(schooling_2, prediction_model_schooling_2,
                       x_data_column_name_schooling_2,
                       y_data_column_name_schooling_2, year_from, year_to,
                       'schooling_2_filled_year_data_set')


'''
    SCHOOLING 3  DATA SET
'''

x_data_column_name_schooling_3 = ['Country Code', 'Year']
y_data_column_name_schooling_3 = ['School enrollment, primary (% gross)']

schooling_3, x_data_schooling_3, y_data_schooling_3, x_data_encoded_schooling_3, \
x_train_schooling_3, x_test_schooling_3, y_train_schooling_3, y_test_schooling_3 = get_important_data(
    documents_destination + "schooling.xlsx",
    "exel", x_data_column_name_schooling_3, y_data_column_name_schooling_3)

schooling_3 = schooling_3[['Country Code', 'Year', 'School enrollment, primary (% gross)']]
prediction_model_schooling_3 = predict_data_set(x_train_schooling_3,
                                                   x_test_schooling_3,
                                                   y_train_schooling_3,
                                                   y_test_schooling_3,
                                                   'SCHOOLING_3 DATA SET', 'mean',
                                                   'elastic')

fill_missing_year_data(schooling_3, prediction_model_schooling_3,
                       x_data_column_name_schooling_3,
                       y_data_column_name_schooling_3, year_from, year_to,
                       'schooling_3_filled_year_data_set')


'''
military_expenditure_formated
'''

