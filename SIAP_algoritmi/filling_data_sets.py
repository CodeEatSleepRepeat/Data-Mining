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

    return new_data_set


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
            if row[1] in country_to_missing_years_map[row[0]]:
                country_to_missing_years_map[row[0]].remove(row[1])

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
    return fill_data_sets(prediction_model, x_data_encoded_full, x_data_full, y_data_full, file_name)


def make_filled_table(destination_from, doc_type, destination_to, data_set_name, x_column_names, y_column_name,
                      prediction_alg_name, nan_filling_method, start_year, end_year):
    x_data_column_name = x_column_names
    y_data_column_name = y_column_name

    data_set, x_data, y_data, x_data_encoded, \
    x_train, x_test, y_train, y_test = get_important_data(
        destination_from,
        doc_type, x_data_column_name, y_data_column_name)

    data_set = data_set[x_column_names + y_column_name]
    prediction_model = predict_data_set(x_train,
                                        x_test,
                                        y_train,
                                        y_test,
                                        data_set_name, nan_filling_method,
                                        prediction_alg_name)

    return fill_missing_year_data(data_set, prediction_model,
                                  x_data_column_name,
                                  y_data_column_name, start_year, end_year,
                                  destination_to)


'''
    - prvo pokupim sve nazive kolona.
    - instaciraj i=1
    - kreni for petlju kroz kolone i za svaku instaciraj data_set
    - na svakom sledecem samo kokanteniramo prediktovanu kolonu kolonu...
'''


def make_filled_table_multi_columns(destination_from, doc_type, destination_to, data_set_name,
                                    x_column_names,
                                    prediction_alg_name, nan_filling_method, year_from, year_to):
    if doc_type == "exel":
        document = utils.read_exel(destination_from)
    else:
        document = utils.read_csv(destination_from)

    column_names_all = document.columns
    final_data_set = []
    iterator = 0

    column_names = []

    # vadimo nazive kolona koje trebamo da predvidimo
    for column_name in column_names_all:
        if column_name in x_column_names:
            continue
        else:
            column_names.append(column_name)

    # pravimo niz brojeva kolona koje se koriste za X vrednost i broj koji se koristi za y vrednost
    column_number = 1
    x_column_number_identificator = []
    for column_x in x_column_names:
        x_column_number_identificator.append(str(column_number))
        column_number = column_number + 1
    x_column_number_identificator.append(str(column_number))

    for column_name in column_names:
        y_column_name = [column_name]

        filled_column_data_set = make_filled_table(destination_from, doc_type, destination_to, data_set_name,
                                                   x_column_names, y_column_name,
                                                   prediction_alg_name, nan_filling_method, year_from, year_to)
        if iterator == 0:
            final_data_set = filled_column_data_set
        else:
            filled_column_data_set = pd.DataFrame(filled_column_data_set, columns=x_column_number_identificator)
            final_data_set = np.concatenate((final_data_set, filled_column_data_set[[str(column_number)]].values), axis=1)

        iterator = iterator + 1

    save_data_as_csv(final_data_set, destination_to_save_new_documents + destination_to)


'''
    SHARED FIELDS:
'''
year_from = 1990
year_to = 2017
documents_destination = "/home/sale/Desktop/GitHubSIAP/Data-Mining/josTrebaRegresija/novo/novo2/"
destination_to_save_new_documents = "/home/sale/PycharmProjects/fillovane_tabele/"

'''
    MAIN
    DATA SETOVI
'''

'''
AGRICULTURAL METHANE EMISSION DATA SET
'''

# destination_from = documents_destination + "agricultural_methane_emission_filled_data_set.xlsx"
# doc_type = "exel"
# destination_to = 'agricultural_methane_emission_filled_years_data_set'
# data_set_name = "AGRICULTURAL METHANE EMISSION DATA SET"
# x_column_names = ['Country Code', 'Year']
# y_column_name = ['Agricultural methane emissions (% of total)']
# prediction_alg_name = "elastic"
# nan_filling_method = "mean"
#
# make_filled_table(destination_from, doc_type, destination_to, data_set_name, x_column_names, y_column_name,
#                   prediction_alg_name, nan_filling_method, year_from, year_to)


'''
CORRUPTION PERCEPTION INDEX 2 DATASET
'''
# destination_from = documents_destination + "coruption_perception_index_2_filled_data_set.xlsx"
# doc_type = "exel"
# destination_to = 'coruption_perception_index_2_filled_years'
# data_set_name = "CORRUPTION PERCEPTION INDEX 2 DATASET"
# x_column_names = ['Country Code', 'Year']
# y_column_name = ['corruption_index']
# prediction_alg_name = "elastic"
# nan_filling_method = "mean"
#
# make_filled_table(destination_from, doc_type, destination_to, data_set_name, x_column_names, y_column_name,
#                   prediction_alg_name, nan_filling_method, year_from, year_to)

'''
DAILY PER CAPITA SUPPLY OF CALORIES DATA SET
'''
# destination_from = documents_destination + "daily_per_capita_supply_of_calories_filled_years_data_set.xlsx"
# doc_type = "exel"
# destination_to = 'daily_per_capita_supply_of_calories_filled_years_data_set'
# data_set_name = "DAILY PER CAPITA SUPPLY OF CALORIES DATA SET"
# x_column_names = ['Country Code', 'Year']
# y_column_name = ['Daily caloric supply (kcal/person/day)']
# prediction_alg_name = "elastic"
# nan_filling_method = "mean"
#
# make_filled_table(destination_from, doc_type, destination_to, data_set_name, x_column_names, y_column_name,
#                   prediction_alg_name, nan_filling_method, year_from, year_to)


'''
LIFE SATISFACTION DATA SET
'''
# destination_from = documents_destination + "life_satisfaction_filled_years_data_set.xlsx"
# doc_type = "exel"
# destination_to = 'life_satisfaction_filled_years_data_set'
# data_set_name = "LIFE SATISFACTION DATA SET"
# x_column_names = ['Country Code', 'Year']
# y_column_name = ['Happines']
# prediction_alg_name = "elastic"
# nan_filling_method = "mean"
#
# make_filled_table(destination_from, doc_type, destination_to, data_set_name, x_column_names, y_column_name,
#                   prediction_alg_name, nan_filling_method, year_from, year_to)


'''
POLITICAL REGIME UPDATED2016 DATA SET
'''
# destination_from = documents_destination + "political_regime_updated2016_filled_years_data_set.xlsx"
# doc_type = "exel"
# destination_to = 'political_regime_updated2016_filled_years_data_set'
# data_set_name = "POLITICAL REGIME UPDATED2016 DATA SET"
# x_column_names = ['Country Code', 'Year']
# y_column_name = ['Political Regime (OWID based on Polity IV and Wimmer & Min) (Score)']
# prediction_alg_name = "elastic"
# nan_filling_method = "mean"
#
# make_filled_table(destination_from, doc_type, destination_to, data_set_name, x_column_names, y_column_name,
#                   prediction_alg_name, nan_filling_method, year_from, year_to)


'''
SHARE WITH ALCOHOL OR DRUG USE DISORDERS
'''
# destination_from = documents_destination + "share-with-alcohol-or-drug-use-disorders.xlsx"
# doc_type = "exel"
# destination_to = 'share_with_alcohol_or_drug_use_disorders_filled_years_data_set'
# data_set_name = "SHARE WITH ALCOHOL OR DRUG USE DISORDERS"
# x_column_names = ['Country Code', 'Year']
# y_column_name = ['Prevalence - Alcohol and substance use disorders: Both (age-standardized percent) (%)']
# prediction_alg_name = "elastic"
# nan_filling_method = "mean"
#
# make_filled_table(destination_from, doc_type, destination_to, data_set_name, x_column_names, y_column_name,
#                   prediction_alg_name, nan_filling_method, year_from, year_to)

'''
UN MIGRANT STOCK BY ORIGIN AND DESTINATION 2019
'''
# destination_from = documents_destination + "UN_MigrantStockByOriginAndDestination_2019.xlsx"
# doc_type = "exel"
# destination_to = 'UN_MigrantStockByOriginAndDestination_2019_filled_year_data_set'
# data_set_name = "UN MIGRANT STOCK BY ORIGIN AND DESTINATION 2019"
# x_column_names = ['Country Code', 'Year']
# y_column_name = ['Total origin']
# prediction_alg_name = "elastic"
# nan_filling_method = "mean"
#
# make_filled_table(destination_from, doc_type, destination_to, data_set_name, x_column_names, y_column_name,
#                   prediction_alg_name, nan_filling_method, year_from, year_to)


'''
FREEDOM OF PRESS DATA, LEGAL ENVIRONMENT AS PREDICTING VALUE
'''
# destination_from = documents_destination + "Freedom_of_the_Press_Data.xlsx"
# doc_type = "exel"
# destination_to = 'freedom_of_press_data_filled_total_score_year_data_set'
# data_set_name = "FREEDOM OF PRESS DATA DATA SET"
# x_column_names = ['Country Code', 'Year']
# y_column_name = ['Total Score']
# prediction_alg_name = "elastic"
# nan_filling_method = "zero"
#
# make_filled_table(destination_from, doc_type, destination_to, data_set_name, x_column_names, y_column_name,
#                   prediction_alg_name, nan_filling_method, year_from, year_to)


'''
    EMPLOYERS PERCENTAGE
'''
# destination_from = documents_destination + "employers_percentage.xlsx"
# doc_type = "exel"
# destination_to = 'employers_percentage_filled_year_data_set'
# data_set_name = "EMPLOYERS PERCENTAGE"
# x_column_names = ['Country Code', 'Year']
# y_column_name = ['Employers, total (% of total employment) (modeled ILO estimate)']
# prediction_alg_name = "elastic"
# nan_filling_method = "mean"
#
# make_filled_table(destination_from, doc_type, destination_to, data_set_name, x_column_names, y_column_name,
#                   prediction_alg_name, nan_filling_method, year_from, year_to)


'''
    inflation data set
'''
# destination_from = documents_destination + "inflation.xlsx"
# doc_type = "exel"
# destination_to = 'inflation_data_set_filled_year_data_set'
# data_set_name = "INFLATION DATA SET"
# x_column_names = ['Country Code', 'Year']
# y_column_name = ['inflation']
# prediction_alg_name = "elastic"
# nan_filling_method = "mean"
#
# make_filled_table(destination_from, doc_type, destination_to, data_set_name, x_column_names, y_column_name,
#                   prediction_alg_name, nan_filling_method, year_from, year_to)


'''
   Population density data set 
'''
# destination_from = documents_destination + "population_density.xlsx"
# doc_type = "exel"
# destination_to = 'population_density_filled_year_data_set'
# data_set_name = "POPULATION DENSITY DATA SET"
# x_column_names = ['Country Code', 'Year']
# y_column_name = ['Population density (people per sq. km of land area)']
# prediction_alg_name = "elastic"
# nan_filling_method = "mean"
#
# make_filled_table(destination_from, doc_type, destination_to, data_set_name, x_column_names, y_column_name,
#                   prediction_alg_name, nan_filling_method, year_from, year_to)

'''
    PTS-2019 DATA SET
'''
# destination_from = documents_destination + "PTS-2019x.xlsx"
# doc_type = "exel"
# destination_to = 'pts_2019_filled_year_data_set'
# data_set_name = "PTS-2019 DATA SET"
# x_column_names = ['Country Code', 'Year']
# y_column_name = ['PTS']
# prediction_alg_name = "elastic"
# nan_filling_method = "mean"
#
# make_filled_table(destination_from, doc_type, destination_to, data_set_name, x_column_names, y_column_name,
#                   prediction_alg_name, nan_filling_method, year_from, year_to)


'''
    UNEPLOYMENT DATA
'''
# destination_from = documents_destination + "unemployment.xlsx"
# doc_type = "exel"
# destination_to = 'uneployment_filled_year_data_set'
# data_set_name = "UNEPLOYMENT DATA SET"
# x_column_names = ['Country Code', 'Year']
# y_column_name = ['Unemployment, total (% of total labor force) (national estimate)']
# prediction_alg_name = "elastic"
# nan_filling_method = "mean"
#
# make_filled_table(destination_from, doc_type, destination_to, data_set_name, x_column_names, y_column_name,
#                   prediction_alg_name, nan_filling_method, year_from, year_to)


'''
    URBAN POPULATION GROWTH 
'''
# destination_from = documents_destination + "urban_population_growth_sredjeno.xlsx"
# doc_type = "exel"
# destination_to = 'urban_population_growth_filled_year_data_set'
# data_set_name = "URBAN POPULATION DATA SET"
# x_column_names = ['Country Code', 'Year']
# y_column_name = ['Urban population growth(%)']
# prediction_alg_name = "elastic"
# nan_filling_method = "mean"
#
# make_filled_table(destination_from, doc_type, destination_to, data_set_name, x_column_names, y_column_name,
#                   prediction_alg_name, nan_filling_method, year_from, year_to)


'''
internet_users_filtered
'''
# destination_from = documents_destination + "internet_users_filtered.xlsx"
# doc_type = "exel"
# destination_to = 'internet_users_filled_year_data_set'
# data_set_name = "INTERNET USERS DATA SET"
# x_column_names = ['Country Code', 'Year']
# y_column_name = ['usage']
# prediction_alg_name = "elastic"
# nan_filling_method = "mean"
#
# make_filled_table(destination_from, doc_type, destination_to, data_set_name, x_column_names, y_column_name,
#                   prediction_alg_name, nan_filling_method, year_from, year_to)


'''
military_expenditure_formated
'''
# destination_from = documents_destination + "military_expenditure_formated.xlsx"
# doc_type = "exel"
# destination_to = 'military_expenditure_formated_year_data_set'
# data_set_name = "MILITARY EXPENDITURE SET"
# x_column_names = ['Country Code', 'Year']
# prediction_alg_name = "elastic"
# nan_filling_method = "mean"
#
# make_filled_table_multi_columns(destination_from, doc_type, destination_to, data_set_name,
#                                     x_column_names,
#                                     prediction_alg_name, nan_filling_method, year_from, year_to)


'''
school_enrollment
'''
# destination_from = documents_destination + "school_enrollment.xlsx"
# doc_type = "exel"
# destination_to = 'school_enrollment_formated_year_data_set'
# data_set_name = "SCHOOL ENROLLMENT DATA SET"
# x_column_names = ['Country Code', 'Year']
# prediction_alg_name = "elastic"
# nan_filling_method = "mean"
#
# make_filled_table_multi_columns(destination_from, doc_type, destination_to, data_set_name,
#                                     x_column_names,
#                                     prediction_alg_name, nan_filling_method, year_from, year_to)

'''
Global_State_of_Democracy_Dataset
'''
# destination_from = documents_destination + "Global_State_of_Democracy_Dataset.xlsx"
# doc_type = "exel"
# destination_to = 'Global_State_of_Democracy_Dataset_formated_year_data_set'
# data_set_name = "GLOBAL STATE OF DEMOCRACY DATA SET"
# x_column_names = ['Country Code', 'Year']
# prediction_alg_name = "elastic"
# nan_filling_method = "mean"
#
# make_filled_table_multi_columns(destination_from, doc_type, destination_to, data_set_name,
#                                     x_column_names,
#                                     prediction_alg_name, nan_filling_method, year_from, year_to)

'''
Freedom of press data filled year data set
'''
# destination_from = documents_destination + "freedom_of_press_data_filled_year_data_set.xlsx"
# doc_type = "exel"
# destination_to = 'freedom_of_press_data_filled_year_data_set_2'
# data_set_name = "FREEDOM OF PRESS DATA"
# x_column_names = ['Country Code', 'Year']
# prediction_alg_name = "elastic"
# nan_filling_method = "mean"
#
# make_filled_table_multi_columns(destination_from, doc_type, destination_to, data_set_name,
#                                     x_column_names,
#                                     prediction_alg_name, nan_filling_method, year_from, year_to)

'''
Human rights scores
'''
# destination_from = documents_destination + "human-rights-scores.xlsx"
# doc_type = "exel"
# destination_to = 'human-rights-scores_filled_year_data_set'
# data_set_name = "HUMAN RIGHTS DATA SET"
# x_column_names = ['Country Code', 'Year']
# y_column_name = ['Human Rights Protection Scores']
# prediction_alg_name = "elastic"
# nan_filling_method = "mean"
#
# make_filled_table(destination_from, doc_type, destination_to, data_set_name, x_column_names, y_column_name,
#                   prediction_alg_name, nan_filling_method, year_from, year_to)

'''
Schooling DATASET
'''

# destination_from = documents_destination + "schooling_1_filled_year_data_set.xlsx"
# doc_type = "exel"
# destination_to = 'schooling_1_filled_year_data_set_formated_year_data_set'
# data_set_name = "SCHOOLING DATASET"
# x_column_names = ['Country Code', 'Year']
# prediction_alg_name = "elastic"
# nan_filling_method = "mean"
#
# make_filled_table_multi_columns(destination_from, doc_type, destination_to, data_set_name,
#                                     x_column_names,
#                                     prediction_alg_name, nan_filling_method, year_from, year_to)

'''
ELECTICITY 
'''
# destination_from = documents_destination + "electricity.xlsx"
# doc_type = "exel"
# destination_to = 'electricity_filled_year_data_set'
# data_set_name = "ELECTRICITY DATA SET"
# x_column_names = ['Country Code', 'Year']
# y_column_name = ['Access to electricity (% of population)']
# prediction_alg_name = "elastic"
# nan_filling_method = "mean"
#
# make_filled_table(destination_from, doc_type, destination_to, data_set_name, x_column_names, y_column_name,
#                   prediction_alg_name, nan_filling_method, year_from, year_to)

'''
GLOBAL MEDIA FREEDOM DATASET
'''
# destination_from = documents_destination + "Global_Media_Freedom_Dataset.xlsx"
# doc_type = "exel"
# destination_to = 'Global_Media_Freedom_Dataset_filled_year_data_set'
# data_set_name = "GLOBAL MEDIA FREEDOM DATA SET"
# x_column_names = ['Country Code', 'Year']
# y_column_name = ['Mediascore']
# prediction_alg_name = "elastic"
# nan_filling_method = "mean"
#
# make_filled_table(destination_from, doc_type, destination_to, data_set_name, x_column_names, y_column_name,
#                   prediction_alg_name, nan_filling_method, year_from, year_to)


'''
FINAL DATASET
'''
destination_from = documents_destination + "elasticNetTabela.csv"
doc_type = "csv"
destination_to = 'final_dataset_filled_year_data_set_formated_year_data_set'
data_set_name = "FINAL DATASET"
x_column_names = ['Country Code', 'Year', 'Country Name', 'Region', 'IncomeGroup', 'Dominant religion',
                  'Deaths - Self-harm - Sex: Both - Age: All Ages (Percent) (%)']
prediction_alg_name = "elastic"
nan_filling_method = "mean"

make_filled_table_multi_columns(destination_from, doc_type, destination_to, data_set_name,
                                x_column_names,
                                prediction_alg_name, nan_filling_method, year_from, year_to)
