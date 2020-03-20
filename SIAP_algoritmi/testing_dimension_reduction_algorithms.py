import dimension_reduction_algorithms
import prediction_algorithms
import validation_methods
import utils
import pandas as pd


'''
COLLECTING DATA SET FROM URL AND SPLITTING TO TRAIN/TEST SETS
'''
collected_data = utils.read_exel("/home/sale/Desktop/GitHubSIAP/Data-Mining/SIAP_algoritmi/military_expenditure_formated.xlsx")
# data_set = collected_data[['Country Code', 'Year', 'Human Rights Protection Scores']]
# x_train, x_test, y_train, y_test = utils.train_test_split_data(data_set[:, 0:2], data_set[:, 2], 0.2)
x_train, x_test, y_train, y_test = utils.train_test_split_data(collected_data[['Country Code', 'Year', 'Military expenditure (current USD)', 'Military expenditure (% of general government expenditure)']],
                                                               collected_data[['Military expenditure (% of GDP)']], 0.2)
# print(x_train.head())

'''
MISSING VALUE RATIO

sa drugim parametrom mu napominjes koliki treshold zelis, sto znaci ukoliko je 0,6 onda 60%+ podataka treba da bude
NaN vrednost kako bi se kolona izbacila
'''
variables = dimension_reduction_algorithms.missing_value_ratio(x_train, 0.6)
# print(variables)


'''
LOW VARIANCE FILTER
drugi parametar predstavlja treshold koji oznacava % koliko varijacije je potrebno u podacima u koloni
da ima kako bi kolona opstala
'''
variables = dimension_reduction_algorithms.low_variance_filter(x_train, 80)
# print(variables)


'''
RANDOM FOREST

zahteva da sve vrednosti budu numericke i da ne budu NaN ili infinity
on odma uradi one hot  encoding za categoricke vrednosti i sve so treba
Ako ukljucimo graph on da lepo prikaz koje kolone uticu na predvidjanje vrednosti
Povratna vrednost je malo cudnolika, ali vraca nekakvu matricu (mozda redukovanu nasu)
'''
x_train_without_nan = x_train.fillna(x_train.mean())
y_train_without_nan = y_train.fillna(y_train.mean())
# random_forest_data_set = dimension_reduction_algorithms.random_forest(x_train_without_nan, y_train_without_nan, False, 1, 10)
# print(random_forest_data_set)


'''
Principal Component Analysis PCA

trazi da sve vrednosti budu numericke, te sam uradio one hot encoding od x_train inline u pozivu funkcije
ne prihvata NaN vrednosti

vraca dataset sa 6 kolona kao sto smo naveli da zelimo da ima, valjda je uradio neku redukciju :D
'''
x_train_without_nan = x_train.fillna(x_train.mean())
# pca_data_set = dimension_reduction_algorithms.principal_component_analysis_pca(pd.get_dummies(x_train_without_nan), 6)
# print(pca_data_set)


'''
FACTOR ANALYSIS

zahteva da sve vrednosti budu numericke, te sam uradio one hot encoding od x_train inline u pozivu funkcije
ne prihvata NaN vrednosti

stampanje grafikona pravi neku gresku treba jos ispitati
kao rezultat vratio je matricu sa brojem kolona koliko smo mi prosledili za n_components
'''
# factor_analysis = dimension_reduction_algorithms.factor_analysis(pd.get_dummies(x_train_without_nan), 20, False)
# print(factor_analysis.shape)


'''
BACKWARD FEATURE SELECTION

Ne prihvata stringove u tabeli, znaci samo brojeve, one hot encoding od x_train.. ali uz to kao i svi drugi 
zahteva da nema NaN vrednosti...

iz nekog razloga vraca dataset sa svim nulama
'''
# backward_feature_selection_data_set = dimension_reduction_algorithms.backward_feature_selection(pd.get_dummies(x_train_without_nan), y_train_without_nan, 6)
# print(backward_feature_selection_data_set)


'''
FORWARD FEATURE SELECTION

Trazi da dataset nema kategorijske vrednosti niti NaN vrednosti..
Vraca kao rezultat nazive kolona koje je on izabrao..
'''
# forward_feature_selection_data_set = dimension_reduction_algorithms.forward_feature_selection(pd.get_dummies(x_train_without_nan), y_train_without_nan, 0)
# print(forward_feature_selection_data_set)

'''
AUTO ENCODER

kao rezultat vraca nekakav dataset sa brojem kolona koje smo mi prosledili u funkciju
'''
# auto_encoder_data_set = dimension_reduction_algorithms.autoencoder_dimension_reduction(pd.get_dummies(x_train_without_nan), 10)
# print(auto_encoder_data_set.shape)


'''
LINEAR DISCRIMINANT ANALYSIS

Dobijam error : ValueError: Unknown label type: (array([0.64953926, 2.00401235, 2.1730623 , ..., 2.79262667, 2.79262667,
       2.79262667]),) 
       
       u pozivu .fit funkcije ovog algoritma
'''
# linear_discriminant_analysis_data_set = dimension_reduction_algorithms.linear_discriminant_analysis(pd.get_dummies(x_train_without_nan), y_train_without_nan, 10)


'''
IDEPEDENT COMPONENT ANALYSIS - ICA

vraca dataset sa brojem kolona (feature-a) koliko smo mu prosledili u funkciju
'''
# ica_data_set = dimension_reduction_algorithms.idepedent_component_analysis_ica(pd.get_dummies(x_train_without_nan), 10, False)
# print(ica_data_set.shape)

'''
UNIFORM MANIFOLD APPROXIMATION AND PROJECTION

ok i ovo izbacuje dataset sa izabranim brojem kolona, ali izbacuje i neke warninge pre toga u crvenoj boji...
'''
# umap_data_set = dimension_reduction_algorithms.uniform_manifold_approximation_and_projection(pd.get_dummies(x_train_without_nan), 10, False)
# print(umap_data_set.shape)

'''
LASSO REGULARISATION 

ima neku gresku
'''
# lasso_data_set = dimension_reduction_algorithms.lasso_regularisation_reduction_dimensionality(pd.get_dummies(x_train_without_nan), y_train_without_nan, False)
# print(lasso_data_set.shape)

'''
SEQUENTIAL FEATURE SELECTION
'''
sequential_feat_selection_data_set = dimension_reduction_algorithms.sequential_feature_selection(pd.get_dummies(x_train_without_nan), y_train_without_nan, False)
