import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split
from sklearn import metrics

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=10, stop=300, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 3]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


def regression_train(x_train, y_train):
    """
  Returns trained  regression models
  """
    estimators = dict()

    estimators['svr'] = SVR()
    estimators['svr'].fit(x_train, y_train)

    estimators['rfr'] = RandomForestRegressor()
    estimators['rfr'].fit(x_train, y_train)

    estimators['dtr'] = DecisionTreeRegressor()
    estimators['dtr'].fit(x_train, y_train)

    estimators['lin'] = LinearRegression()
    estimators['lin'].fit(x_train, y_train)

    estimators['knr'] = KNeighborsRegressor()
    estimators['knr'].fit(x_train, y_train)

    return estimators


def regression_scoring(y, yhat):
    """
    Returns regression scores
    """
    scores = dict()
    scores['mae'] = metrics.mean_absolute_error(y, yhat)
    scores['mse'] = metrics.mean_squared_error(y, yhat)
    scores['rmse'] = scores['mse'] ** (0.5)
    # scores['r2'] = metrics.r2_score(y,yhat) # impossible if evaluating only 1 prediction (at least 1 needed)
    return scores


def calculate_weights(errs):
    """
    Revers weights calculation.
    Weight of estimator established this way:
    1) calculate proportion of error for estimator in comparison to sum of errors of estimators
    2) take power of -1 from the proportion
    3) calculate linear equation (a_1 + a_2 +...+ a_n)x=1, where each propotion is "a_i"
    4) calculate weights by multiplying propotions by "x"
    """
    evaluator = dict()
    for est_err in errs.keys():
        evaluator[est_err] = []

    for it in range([len(errs[x]) for x in errs.keys()][0]):
        values = dict()

        for est_err in errs.keys():
            values[est_err] = errs[est_err][it]

        scorers = list()
        for est_err in errs.keys():
            scorers.append((values[est_err] / sum(values.values())) ** (-1))

        x = 1 / sum(scorers)

        for idx, est_err in enumerate(errs.keys()):
            res = x * scorers[idx]
            # if np.isnan(res):
            #  res = 0.0
            evaluator[est_err].append(res)

    return evaluator


def Z_scoring(data, X_values, Y_value, beta):
    """
    
    :param data:
    :param X_values:
    :param Y_value:
    :param beta:
    :return:
    """
    N = int(len(data) / beta)  # chunk size
    finish_ids = [N * x for x in range(beta + 1)[1:]]

    errors = {'svr': [], 'lin': [], 'dtr': [], 'rfr': [],
              'knr': []}  # create dictionary to gather error scores for estimators

    global_test = pd.DataFrame(columns=X_values + list(errors.keys()))
    estimators = None

    for iter in range(beta - 1):  # beta assigned above

        train = data.iloc[:finish_ids[iter]]
        test = data.iloc[finish_ids[iter]:finish_ids[iter + 1]]

        estimators = regression_train(x_train=train[X_values].values, y_train=train[Y_value])  # train estimators

        for idx, row in enumerate(test.iterrows()):

            for est in estimators.keys():
                y_est = estimators[est].predict([test[X_values].iloc[idx].values])  # predict with individual estimator
                errors[est].append(regression_scoring(y=[test[Y_value].iloc[idx]], yhat=y_est)[
                                       'rmse'])  # calculate an error of individual estimator

        global_test = global_test.append(test[X_values], ignore_index=True)

    for est in estimators.keys():
        global_test[est] = errors[est]

    Z_model = dict()
    for est in estimators.keys():
        rf = RandomForestRegressor()
        Z_model[est] = RandomizedSearchCV(estimator=rf, n_iter=10, param_distributions=random_grid,
                                          cv=2, verbose=2, random_state=42,
                                          n_jobs=-1)  # set the random forest regression model for error prediction
        # Z_model[est] = rf
        Z_model[est].fit(global_test[X_values].values, global_test[est])  # train error prediction model
    return Z_model, estimators


def compete(compet_ds, Z_model, estimators, X_values, Y_value):
    """

    :param compet_ds:
    :param Z_model:
    :param estimators:
    :param X_values:
    :param Y_value:
    :return:
    """
    prediction = dict()
    for est in estimators.keys():
        prediction[est] = estimators[est].predict(
            compet_ds[X_values].values)  # calculate the prediction of the estimators

    z_errors = dict()
    for est in Z_model.keys():
        z_errors[est] = Z_model[est].predict(
            compet_ds[X_values].values)  # calculate the prediction of errors for the estimators

    weights = calculate_weights(z_errors)  # obtain weights for estimators

    Z_based_y_hat = np.zeros(shape=(1, len(compet_ds)))
    for est in estimators.keys():
        Z_based_y_hat = np.add(Z_based_y_hat,
                               weights[est] * prediction[est])  # obraining predicted result based on errors
    results = regression_scoring(Z_based_y_hat[0], compet_ds[Y_value])  # compare the prediction with actual values
    return results, Z_based_y_hat
    # Predicted the Errors of the models by Random Forest Regression


def analyze_assembly(learning_ds, competition_ds, X, Y):
    """
    :param learning_ds:
    :param competition_ds:
    :param X:
    :param Y:
    :return:
    """
    Z_regr, predictors = Z_scoring(data=learning_ds, X_values=X, Y_value=Y, beta=20)

    Z_result = compete(compet_ds=competition_ds, Z_model=Z_regr,
                       estimators=predictors, X_values=X, Y_value=Y)
    return Z_result
