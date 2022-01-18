import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LassoLars, ElasticNetCV, BayesianRidge, ARDRegression



def regression_scoring(y, yhat):
    """
    Returns regression scores
    """
    scores = dict()
    #scores['mae'] = metrics.mean_absolute_error(y, yhat)
    scores['squared_error'] = np.square(np.subtract(y, yhat))
    #scores['mse'] = metrics.mean_squared_error(y, yhat)
    #scores['rmse'] = scores['mse'] ** (0.5)
    # scores['r2'] = metrics.r2_score(y,yhat) # impossible if evaluating only 1 prediction (at least 1 needed)
    return scores


def calculate_weights(errs, priority=None, approach=None):
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
    #print(evaluator)

    for it in range([len(errs[x]) for x in errs.keys()][0]):

        values = dict()
        for est_err in errs.keys():
            values[est_err] = errs[est_err][it]
        #print(values)
            

        scorers = list()
        #print(np.exp(list(values.values())))
        
        if approach == "exp":
            tot = np.sum([np.exp(-el) for el in values.values()])
            if tot == 0:
                tot = 1
            #print(tot)
            for est_err in errs.keys():
                scorers.append(np.exp(- values[est_err]) / tot)
        else:
            for est_err in errs.keys():
                #print(values[est_err])

                scorers.append((values[est_err] / sum(values.values())) ** (-1))
            

        #print(scorers)
        if priority:
            min_value = min(scorers)
            min_index = scorers.index(min_value)
            scorers = [0 for x in scorers]
            scorers[min_index] = 1

        #print(scorers)
        if sum(scorers):
            x = 1 / sum(scorers)
        else:
            x = 1

        for idx, est_err in enumerate(errs.keys()):
            res = x * scorers[idx]
            evaluator[est_err].append(res)

    #print(evaluator)
    return evaluator



def Z_scoring(data, X_values, Y_value, beta, estimators):
    """
    
    :param data:
    :param X_values:
    :param Y_value:
    :param beta:
    :return:
    """
    
    N = int(len(data) / beta)  # chunk size
    finish_ids = [N * x for x in range(beta + 1)[1:]]
    print(finish_ids)

    #errors = {'svr': [], 'lin': [], 'dtr': [], 'rfr': []}
    errors = dict((el,[])  for el in estimators) # create dictionary to gather error scores for estimators 
    #print(errors)

    #global_test = pd.DataFrame(columns= X_values + list(errors.keys()))
    global_test = pd.DataFrame(columns=X_values) # + list(errors.keys())
    #print(global_test)
    #estimators = init_estimators()  #innitialize estimators here
    #estimators = est # take previously trained estimators
    
    

    for iter in range(beta-1):  # beta assigned above

        # If training estimators
        train = data.iloc[:finish_ids[iter]]
        test = data.iloc[finish_ids[iter]:finish_ids[iter + 1]]
        estimators = train_estimators(estimators, x_train=train[X_values].values, y_train=train[Y_value].values)  # train estimators
        
        # Without estimators training
        test = data.iloc[0:finish_ids[iter]]
        #print(iter, test)

        #estimators = regression_train(x_train=train[X_values].values, y_train=train[Y_value])  # train estimators

        for est in estimators.keys():
            y_est = estimators[est].predict(test[X_values].values)  # predict with individual estimations
            errors[est] += regression_scoring(y=test[Y_value].values, yhat=y_est)['squared_error'].tolist()  # calculate an error of individual estimator
            #print(errors[est]) # HEREREREREE
            

        global_test = global_test.append(test[X_values], ignore_index=True)
        

    
    for k in errors.keys():
        #print(k)
        #print(errors[k])
        global_test[k] = errors[k][0]
        
    #print(global_test)
    
    #global_test.to_csv("CHECH.CSV")
    #print(global_test)
    

    Z_model = dict()
    for est in estimators.keys():
        rf = RandomForestRegressor()
        
        param_grid = {
        'bootstrap': [True],
        'max_depth': [50, 80, 100, 110],
        'max_features': [2, 3, 5, 10],
        'min_samples_leaf': [2, 3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [10, 20, 50, 100, 200]
                      }
        
        
        Z_model[est] = RandomizedSearchCV(estimator=rf, n_iter=10, param_distributions=param_grid,
                                         cv=2, random_state=42,
                                         n_jobs=-1)  # set the random forest regression model for error prediction
        
        
        
        Z_model[est].fit(global_test[X_values].values, global_test[est].values)  # train error prediction model
    return Z_model, estimators




def compete(compet_ds, Z_model, estimators, X_values, Y_value, priority=None, appr=None):
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

    weights = calculate_weights(z_errors, priority=priority, approach=appr)  # Set to obtain the weights for estimators WITH/WITHOUT PRIORITY

    Z_based_y_hat = np.zeros(shape=(1, len(compet_ds)))
    for est in estimators.keys():
        #print(Z_based_y_hat)
        Z_based_y_hat = np.add(Z_based_y_hat, weights[est] * prediction[est]) # obtaining predicted result based on errors

    results = regression_scoring(Z_based_y_hat[0], compet_ds[Y_value])
    return results
    # Predicted the Errors of the models by Random Forest Regression
    

def init_est():
        """
        Returns trained  regression models 
        """
        estimators = dict()
        
        estimators['svr'] = SVR()
        params = {'epsilon':[0.1, 0.01, 0.2, 0.5]}
    
        estimators['svr'] = GridSearchCV(estimators['svr'], params)
     
        estimators['rfr'] = RandomForestRegressor()
        params = {'max_depth':[1, 1.5, 0.5, 0.1, 2, 5]}
        estimators['rfr'] = GridSearchCV(estimators['rfr'], params)

        estimators['dtr'] = DecisionTreeRegressor()
        #params = {'criterion':['squared_error', 'friedman_mse', 'absolute_error']}
        params = {'max_depth':[1, 1.5, 0.5, 0.1, 2, 5]}
        estimators['dtr'] = GridSearchCV(estimators['dtr'], params)

        estimators['lin'] = LinearRegression()
        params = {'fit_intercept':[True, False]}
        estimators['lin'] = GridSearchCV(estimators['lin'], params)
        
        estimators['lasso'] = LassoLars(alpha=.1, normalize=False) #Least Angle Regression
        params = {'fit_intercept':[True, False]}
        estimators['lasso'] = GridSearchCV(estimators['lasso'], params)
        
        estimators['elastic'] = ElasticNetCV()
        estimators['elastic'] = GridSearchCV(estimators['elastic'], params)
    
        estimators['bayes'] = BayesianRidge()
        estimators['bayes'] = GridSearchCV(estimators['bayes'], params)

        estimators['ARD'] = ARDRegression()
        estimators['ARD'] = GridSearchCV(estimators['ARD'], params)

        return estimators
    
def train_estimators(estimators, x_train, y_train):
        """
        Returns trained  regression models 
        """
        estimators['svr'].fit(x_train, y_train)
        estimators['rfr'].fit(x_train, y_train)
        estimators['dtr'].fit(x_train, y_train)
        estimators['lin'].fit(x_train, y_train)
        estimators['lasso'].fit(x_train, y_train)
        estimators['elastic'].fit(x_train, y_train)
        estimators['bayes'].fit(x_train, y_train)
        estimators['ARD'].fit(x_train, y_train)

        return estimators
    


def analyze_ensemble(learning_ds, competition_ds, X, Y,  trained_est = None, priority=None, approach=None, beta=None):
    """
    :param learning_ds:
    :param competition_ds:
    :param X:
    :param Y:
    :return:
    """
    if not trained_est:
        est = init_est()
    else:
        est = trained_est
    
    print("Beta " + str(beta))
    
    Z_regr, predictors = Z_scoring(data=learning_ds, X_values=X, Y_value=Y, beta=beta, estimators = est)
    
    print("Part Completed")

    Z_result = compete(compet_ds=competition_ds, Z_model=Z_regr,
                       estimators=predictors, X_values=X, Y_value=Y, priority=priority, appr=approach)
    return Z_result
