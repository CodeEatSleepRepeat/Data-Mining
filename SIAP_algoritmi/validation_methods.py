from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import cross_val_predict


def r2_metrics(y_true, y_predicted):
    return r2_score(y_true, y_predicted)


def mean_squared_error_metrics(y_true, y_predicted):
    return mean_squared_error(y_true, y_predicted)


def accuracy_score_metrics(y_true, y_predicted):
    return accuracy_score(y_true,y_predicted)


def cross_validation(model, x, y, cv):
    y_cv = cross_val_predict(model, x, y, cv=10)
    return y_cv


'''
We can also use K-Fold Cross-Validation with the Cox Model and the Aalen Additive Model. 
The function splits the data into a training set and a testing set and fits itself on the training set and evaluates itself on the testing set. 
The function repeats this for each fold.

def k_fold_cross_validation_metrics(model, dataset):
    scores = k_fold_cross_validation(model, dataset, 'T', event_col='E', k=10)
    return scores
    
nisu dali bas dobar kod za ovo, tako da cu morati jos da istrazim
'''
