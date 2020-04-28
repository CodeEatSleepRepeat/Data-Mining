import dimension_reduction_algorithms
import prediction_algorithms
import validation_methods
import utils
import pandas as pd

'''
    Funkcija koja nam automatski izdvaja sve X kolone od Y kolone i vraca
    ta 2 skupa nazad kao povratnu vrednost
'''


def get_x_y_column_names(data_set, y_name):
    column_names_all = data_set.columns
    x_column_names = []
    y_column_name = []

    for column_name in column_names_all:
        if column_name == y_name:
            y_column_name.append(column_name)
        else:
            x_column_names.append(column_name)

    return x_column_names, y_column_name


'''
    Funkcija koja racuna i stampa rezultate treniranih modela
'''


def print_error_results(y_true, y_predicted, data_set_name, prediction_algorithm_name,
                        dimension_reduction_algorithm_name):
    print("#########################################################")
    print(
        "Test results for " + data_set_name + " reduced with " + dimension_reduction_algorithm_name + " with training " +
        " model " + prediction_algorithm_name)

    MSE = validation_methods.mean_squared_error_metrics(y_true, y_predicted)
    r2_error = validation_methods.r2_metrics(y_true, y_predicted)
    print("MSE: ")
    print(MSE)
    print("R2 ERROR: ")
    print(r2_error)


'''
    Funkcija koja na osnovu ulaznih parametara pravi predikcioni
    x boost model, testira i stampa njegove rezultate
'''


def predict_with_x_boost(x_train, y_train, x_test, y_test, data_set_name, dimension_reduction_name):
    x_boost_prediction_model = prediction_algorithms.XGBoost_regression(x_train, y_train)
    x_boost_prediction = x_boost_prediction_model.predict(x_test)
    print_error_results(y_test, x_boost_prediction, data_set_name, "XBoost regression",
                        dimension_reduction_name)


'''
    Funkcija koja na osnovu ulaznih parametara pravi predikcioni
    elastic net model, testira i stampa njegove rezultate
'''


def predict_with_elastic_net(x_train, y_train, x_test, y_test, data_set_name, dimension_reduction_name,
                             elastic_net_alpha):
    elastic_net_prediction_model = prediction_algorithms.elastic_regression(x_train, y_train, elastic_net_alpha)
    elastic_net_prediction = elastic_net_prediction_model.predict(x_test)
    print_error_results(y_test, elastic_net_prediction, data_set_name, "Elastic net regression",
                        dimension_reduction_name)


'''
    Funkcija koja je sluzila da na osnovu prosledjenih naziva kolona izdvoji x test i x train
'''


def get_x_train_test_random_forest_elastic_net(data_set, column_names, x_train, x_test):
    x_train_pd = pd.DataFrame(x_train, columns=data_set.columns)
    x_test_pd = pd.DataFrame(x_test, columns=data_set.columns)
    x_train_reduced = x_train_pd[column_names]
    x_test_reduced = x_test_pd[column_names]
    return x_train_reduced, x_test_reduced


'''
    Funkcija koja na osnovu tresholda vraca x_train i x_test preko redukcionog algoritma
    missing value ratio
    
    data_set_name se koristio u predhodnoj verziji, pa da ne bih dirao kod, neka ga za sad haha
'''


def reduce_dimension_missing_value_ratio(data_set, x_train, x_test, threshold, data_set_name):
    reduced_column_names = dimension_reduction_algorithms.missing_value_ratio(x_train, threshold)
    x_train_pd = pd.DataFrame(x_train, columns=data_set.columns)
    x_test_pd = pd.DataFrame(x_test, columns=data_set.columns)
    x_train_reduced = x_train_pd[reduced_column_names]
    x_test_reduced = x_test_pd[reduced_column_names]

    return x_train_reduced, x_test_reduced


'''
    Funkcija koja na osnovu tresholda vraca x_train i x_test preko redukcionog algoritma
    low variance filter
'''


def reduce_dimension_low_variance_filter(data_set, x_train, x_test, threshold, data_set_name):
    reduced_column_names = dimension_reduction_algorithms.low_variance_filter(x_train_e_n, threshold)
    x_train_pd = pd.DataFrame(x_train, columns=data_set.columns)
    x_test_pd = pd.DataFrame(x_test, columns=data_set.columns)
    x_train_reduced = x_train_pd[reduced_column_names]
    x_test_reduced = x_test_pd[reduced_column_names]

    return x_train_reduced, x_test_reduced


'''
    Funkcija koja je sluzila da na osnovu prosledjenih naziva kolona izdvoji x test i x train
'''


def get_x_train_test_random_forest_elastic_net(data_set, column_names, x_train, x_test):
    x_train_pd = pd.DataFrame(x_train, columns=data_set.columns)
    x_test_pd = pd.DataFrame(x_test, columns=data_set.columns)
    x_train_reduced = x_train_pd[column_names]
    x_test_reduced = x_test_pd[column_names]
    return x_train_reduced, x_test_reduced


'''
CONSTANTS
'''
elastic_net_final_data_set_location = "../elasticNetTabela.xlsx"
neural_net_final_data_set_location = "../neuralNetworkTabela.xlsx"
prediction_column_name = "Deaths - Self-harm - Sex: Both - Age: All Ages (Percent) (%)"

'''
COLLECTING DATASETS
'''
# ElasticNET full data set
elastic_net_final = utils.read_exel(elastic_net_final_data_set_location)
elastic_net_final = pd.get_dummies(elastic_net_final)
x_column_names_e_n, y_column_names_e_n = get_x_y_column_names(elastic_net_final, prediction_column_name)
x_train_e_n, x_test_e_n, y_train_e_n, y_test_e_n = utils.train_test_split_data(elastic_net_final[x_column_names_e_n],
                                                                               elastic_net_final[y_column_names_e_n],
                                                                               0.2)

# Neural network full data set
neural_net_final = utils.read_exel(neural_net_final_data_set_location)
neural_net_final = pd.get_dummies(neural_net_final)
x_column_names_n_n, y_column_names_n_n = get_x_y_column_names(neural_net_final, prediction_column_name)
x_train_n_n, x_test_n_n, y_train_n_n, y_test_n_n = utils.train_test_split_data(neural_net_final[x_column_names_n_n],
                                                                               neural_net_final[y_column_names_n_n],
                                                                               0.2)

'''
APPLYING DIMENSIONAL REDUCTION ALGORITHMS ON DATA SETS
'''

# Missing value ratio

# en_x_train_m_v_r_reduced, en_x_test_m_v_r_reduced = reduce_dimension_missing_value_ratio(elastic_net_final, x_train_e_n, x_test_e_n, 0.6,
#                                                                                          "Elastic net final")
# nn_x_train_m_v_r_reduced, nn_x_test_m_v_r_reduced = reduce_dimension_missing_value_ratio(neural_net_final, x_train_n_n, x_test_n_n, 0.6,
#                                                                                          "Neural net final dataset")





# Low variance filter
low_variance_threshold = 80

# en_x_train_l_v_f_reduced, en_x_test_l_v_r_reduced = reduce_dimension_low_variance_filter(elastic_net_final, x_train_e_n, x_test_e_n, low_variance_threshold,
#                                                                                          "Elastic net final dataset")
# nn_x_train_l_v_f_reduced, nn_x_test_l_v_r_reduced = reduce_dimension_low_variance_filter(neural_net_final, x_train_n_n, x_test_n_n, low_variance_threshold,
#                                                                                          "Neural net final dataset")





# Random forest

# sa ispod linijom sam dobio graf sa najbitnijim featurima
# nn_x_train_r_f_reduced = dimension_reduction_algorithms.random_forest(x_train_e_n, y_train_e_n, True, 1, 10)

# u ispod liniji sam rucno izvlacio 16 najbitnijih kolona sa grafikona
# en_x_train_r_f_reduced, en_x_test_r_f_reduced = get_x_train_test_random_forest_elastic_net(elastic_net_final,
#                                                                                            ['Fertility rate, total (births per woman)', 'Prevalence - Depressive disorders - Sex: Both - Age: Age-standardized (Percent) (%)', 'Country Code_LKA', 'Country Code_KOR',
#                                                                                             'Prevalence - Alcohol and substance use disorders: Both (age-standardized percent) (%)', 'Prevalence - Mental and substance use disorders - Sex: Both - Age: Age-standardized (Percent) (%)',
#                                                                                             'Population density (people per sq. km of land area)', 'Urban population growth(%)', 'Happines', 'Country Code_SUR', 'Human Rights Protection Scores', 'Per capita CO₂ emissions (tonnes per capita)',
#                                                                                             'Military expenditure (current USD)', 'Employers, total (% of total employment) (modeled ILO estimate)', 'Prevalence - Eating disorders - Sex: Both - Age: Age-standardized (Percent) (%)',
#                                                                                             'Internet usage'], x_train_e_n, x_test_e_n)

# nn_x_train_r_f_reduced = dimension_reduction_algorithms.random_forest(x_train_n_n, y_train_n_n, True, 1, 10)
# nn_x_train_r_f_reduced, nn_x_test_r_f_reduced = get_x_train_test_random_forest_elastic_net(neural_net_final,
#                                                                                            ['Fertility rate, total (births per woman)', 'Prevalence - Depressive disorders - Sex: Both - Age: Age-standardized (Percent) (%)', 'Country Code_LKA', 'Country Code_KOR',
#                                                                                             'Prevalence - Alcohol and substance use disorders: Both (age-standardized percent) (%)', 'Prevalence - Mental and substance use disorders - Sex: Both - Age: Age-standardized (Percent) (%)',
#                                                                                             'Population density (people per sq. km of land area)', 'Happines', 'Per capita CO₂ emissions (tonnes per capita)',
#                                                                                             'Employers, total (% of total employment) (modeled ILO estimate)',
#                                                                                             'Internet usage', 'corruption_index', 'Total origin', 'IncomeGroup_High income', 'Death rate, crude (per 1,000 people)',
#                                                                                             'Agricultural methane emissions (% of total)'], x_train_n_n, x_test_n_n)





# Principal component analysis (PCA) (kao drugi parametar, mi biramo koji cemo broj kolona koje vraca PCA)
pca_number_of_features_en = 187
pca_number_of_features_nn = 168

# en_x_train_pca_reduced = dimension_reduction_algorithms.principal_component_analysis_pca(x_train_e_n, pca_number_of_features_en)
# en_x_test_pca_reduced = dimension_reduction_algorithms.principal_component_analysis_pca(x_test_e_n, pca_number_of_features_en)
#
# nn_x_train_pca_reduced = dimension_reduction_algorithms.principal_component_analysis_pca(x_train_n_n, pca_number_of_features_nn)
# nn_x_test_pca_reduced = dimension_reduction_algorithms.principal_component_analysis_pca(x_test_n_n, pca_number_of_features_nn)





# Factor analysis, kao drugi parametar ide broj kolona na koji zelimo da redukujemo
factor_analysis_number_of_features_en = 5
factor_analysis_number_of_features_nn = 5

# en_x_train_f_a_reduced = dimension_reduction_algorithms.factor_analysis(x_train_e_n, factor_analysis_number_of_features_en, False)
# en_x_test_f_a_reduced = dimension_reduction_algorithms.factor_analysis(x_test_e_n, factor_analysis_number_of_features_en, False)

# nn_x_train_f_a_reduced = dimension_reduction_algorithms.factor_analysis(x_train_n_n, factor_analysis_number_of_features_nn, False)
# nn_x_test_f_a_reduced = dimension_reduction_algorithms.factor_analysis(x_test_n_n, factor_analysis_number_of_features_nn, False)


'''
TRAINING PREDICTION MODELS

Za svaki algoritam bice vise predikcionih modela, u zavisnosti od vrednosti za treniranje:
        - sa originalnim skupom
        - sa redukovanim skupom od strane missing value ratio redukcionog algoritma (moze se i preskociti)
        - sa redukovanim skupom od strane low variance filter redukcionog algoritma
        - sa redukovanim skupom od strane random forest redukcionog algoritma
        - sa redukovanim skupom od strane principal component analysis (PCA) redukcionog algoritma
        - sa redukovanim skupom od strane factor analysis redukcionog algoritma
        
Prvo idu predikcioni modeli za sve kombinacije vezane za Elastic net final dataset, pa onda
za Neural net final dataset
'''



# XBoost regression prediction model

# Elastic net final dataset
# predict_with_x_boost(x_train_e_n, y_train_e_n, x_test_e_n, y_test_e_n, "Elastic net final dataset", "original dataset")
# predict_with_x_boost(en_x_train_m_v_r_reduced, y_train_e_n, en_x_test_m_v_r_reduced, y_test_e_n,
#                      "Elastic net final dataset", "Missing value ratio")
# predict_with_x_boost(en_x_train_l_v_f_reduced, y_train_e_n, en_x_test_l_v_r_reduced, y_test_e_n,
#                      "Elastic net final dataset", "Low variance filter")
# predict_with_x_boost(en_x_train_r_f_reduced, y_train_e_n, en_x_test_r_f_reduced, y_test_e_n, "Elastic net final dataset",
#                      "Random forest")
# predict_with_x_boost(en_x_train_pca_reduced, y_train_e_n, en_x_test_pca_reduced, y_test_e_n, "Elastic net final dataset",
#                      "Principal component analysis")
# predict_with_x_boost(en_x_train_f_a_reduced, y_train_e_n, en_x_test_f_a_reduced, y_test_e_n, "Elastic net final dataset",
#                      "Factor analysis")

# Neural net final dataset
# predict_with_x_boost(x_train_n_n, y_train_n_n, x_test_n_n, y_test_n_n, "Neural net final dataset", "original dataset")
# predict_with_x_boost(nn_x_train_m_v_r_reduced, y_train_n_n, nn_x_test_m_v_r_reduced, y_test_n_n,
#                      "Neural net final dataset", "Missing value ratio")
# predict_with_x_boost(nn_x_train_l_v_f_reduced, y_train_n_n, nn_x_test_l_v_r_reduced, y_test_n_n,
#                      "Neural net final dataset", "Low variance filter")
# predict_with_x_boost(nn_x_train_r_f_reduced, y_train_n_n, nn_x_test_r_f_reduced, y_test_n_n, "Neural net final dataset",
#                      "Random forest")
# predict_with_x_boost(nn_x_train_pca_reduced, y_train_n_n, nn_x_test_pca_reduced, y_test_n_n, "Neural net final dataset",
#                      "Principal component analysis")
# predict_with_x_boost(nn_x_train_f_a_reduced, y_train_n_n, nn_x_test_f_a_reduced, y_test_n_n, "Neural net final dataset",
#                      "Factor analysis")







# Cox regression prediction model (kaze da je prevelik broj kolona za racunanje kolozione matrice...)

# cox_regression_en = prediction_algorithms.cox_regression(elastic_net_final, "Year",
#                                                          prediction_column_name)
# cox_regression_en.print_summary()
#
# cox_regression_nn = prediction_algorithms.cox_regression(neural_net_final, "Year",
#                                                          prediction_column_name)
# cox_regression_nn.print_summary()







# Elastic net regression prediction model
alpha = 0.01

# Elastic net final dataset
# predict_with_elastic_net(x_train_e_n, y_train_e_n, x_test_e_n, y_test_e_n, "Elastic net final dataset",
#                          "original dataset", alpha)
# predict_with_elastic_net(en_x_train_m_v_r_reduced, y_train_e_n, en_x_test_m_v_r_reduced, y_test_e_n, "Elastic net final dataset",
#                          "Missing value ratio", alpha)
# predict_with_elastic_net(en_x_train_l_v_f_reduced, y_train_e_n, en_x_test_l_v_r_reduced, y_test_e_n, "Elastic net final dataset",
#                          "Low variance filter", alpha)
# predict_with_elastic_net(en_x_train_r_f_reduced, y_train_e_n, en_x_test_r_f_reduced, y_test_e_n, "Elastic net final dataset",
#                          "Random forest", alpha)
# predict_with_elastic_net(en_x_train_pca_reduced, y_train_e_n, en_x_test_pca_reduced, y_test_e_n, "Elastic net final dataset",
#                          "Principal component analysis", alpha)
# predict_with_elastic_net(en_x_train_f_a_reduced, y_train_e_n, en_x_test_f_a_reduced, y_test_e_n, "Elastic net final dataset",
#                          "Factor analysis", alpha)

# Neural net final dataset
# predict_with_elastic_net(x_train_n_n, y_train_n_n, x_test_n_n, y_test_n_n, "Neural net final dataset",
#                          "original dataset", alpha)
# predict_with_elastic_net(nn_x_train_m_v_r_reduced, y_train_n_n, nn_x_test_m_v_r_reduced, y_test_n_n, "Neural net final dataset",
#                          "Missing value ratio", alpha)
# predict_with_elastic_net(nn_x_train_l_v_f_reduced, y_train_n_n, nn_x_test_l_v_r_reduced, y_test_n_n, "Neural net final dataset",
#                          "Low variance filter", alpha)
# predict_with_elastic_net(nn_x_train_r_f_reduced, y_train_n_n, nn_x_test_r_f_reduced, y_test_n_n, "Neural net final dataset",
#                          "Random forest", alpha)
# predict_with_elastic_net(nn_x_train_pca_reduced, y_train_n_n, nn_x_test_pca_reduced, y_test_n_n, "Neural net final dataset",
#                          "Principal component analysis", alpha)
# predict_with_elastic_net(nn_x_train_f_a_reduced, y_train_n_n, nn_x_test_f_a_reduced, y_test_n_n, "Neural net final dataset",
#                          "Factor analysis", alpha)







# Partial least squares (PLS)

# Elastic net final dataset
# prediction_algorithms.partial_least_squares(x_train_e_n, y_train_e_n, len(x_train_e_n.columns), True)
# prediction_algorithms.partial_least_squares(en_x_train_m_v_r_reduced, y_train_e_n, len(en_x_train_m_v_r_reduced.columns), True)
# prediction_algorithms.partial_least_squares(en_x_train_l_v_f_reduced, y_train_e_n, len(en_x_train_l_v_f_reduced.columns), True)
# prediction_algorithms.partial_least_squares(en_x_train_r_f_reduced, y_train_e_n, len(en_x_train_r_f_reduced.columns), True)
# prediction_algorithms.partial_least_squares(en_x_train_pca_reduced, y_train_e_n, en_x_train_pca_reduced.shape[1], True)
# prediction_algorithms.partial_least_squares(en_x_train_f_a_reduced, y_train_e_n, en_x_train_f_a_reduced.shape[1], True)

# Neural net final dataset
# prediction_algorithms.partial_least_squares(x_train_n_n, y_train_n_n, len(x_train_n_n.columns), True)
# prediction_algorithms.partial_least_squares(nn_x_train_m_v_r_reduced, y_train_n_n, len(nn_x_train_m_v_r_reduced.columns), True)
# prediction_algorithms.partial_least_squares(nn_x_train_l_v_f_reduced, y_train_n_n, len(nn_x_train_l_v_f_reduced.columns), True)
# prediction_algorithms.partial_least_squares(nn_x_train_r_f_reduced, y_train_n_n, len(nn_x_train_r_f_reduced.columns), True)
# prediction_algorithms.partial_least_squares(nn_x_train_pca_reduced, y_train_n_n, nn_x_train_pca_reduced.shape[1], True)
# prediction_algorithms.partial_least_squares(nn_x_train_f_a_reduced, y_train_n_n, nn_x_train_f_a_reduced.shape[1], True)
