import pandas as pd
import numpy as np

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score





def class_scoring(estim, y, yhat):
    """
    Returns regression scores
    """
    score = list()
    classes = estim.classes_.tolist()
    #print(classes)
    for idx, el in enumerate(y):
        indx_right_class = classes.index(el) # find the index of the right class prediction proba
        score.append(1 - yhat[idx][indx_right_class])

    return score




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


    errors = dict((el,[])  for el in estimators) # create dictionary to gather error scores for estimators 
    
    global_test = pd.DataFrame(columns=X_values) # + list(errors.keys())
    
    
    

    for iter in range(beta-1):  # beta assigned above

        # If training estimators
        train = data.iloc[:finish_ids[iter]]
        test = data.iloc[finish_ids[iter]:finish_ids[iter + 1]]
        #estimators = train_estimators(estimators, x_train=train[X_values].values, y_train=train[Y_value].values)  # train estimators
        
        # Without estimators training
        #test = data.iloc[0:finish_ids[iter]]
        #print(iter, test)

        #estimators = regression_train(x_train=train[X_values].values, y_train=train[Y_value])  # train estimators

        for est in estimators.keys():
            estimators[est].fit(train[X_values].values, train[Y_value].values)
            y_est = estimators[est].predict_proba(test[X_values].values) # predict with individual estimations
            errors[est] = class_scoring(estimators[est], y=test[Y_value].values, yhat=y_est)  # calculate an error of individual estimator
            

        global_test = global_test.append(test[X_values], ignore_index=True)

    
    for k in errors.keys():
        global_test[k] = errors[k][0]
        
    

    Z_model = dict()
    for est in estimators.keys():
        rf = RandomForestRegressor()
        #Z_model[est] = rf # REMOVE
        
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




def compete(compet_ds, Z_model, estimators, X_values, Y_value, priority=None):
    """

    :param compet_ds:
    :param Z_model:
    :param estimators:
    :param X_values:
    :param Y_value:
    :return:
    """
    prediction = dict()
    z_errors = dict()
    for est in estimators.keys():
        prediction[est] = estimators[est].predict(compet_ds[X_values].values)  # calculate the prediction of the estimators # cat dog girrafe
        z_errors[est] = Z_model[est].predict(compet_ds[X_values].values) # calculate the prediction of errors for the estimators # 0.7 0.3 0.4 was wrong
    
    selected_classes = []
    
    if not priority:
        for idx in range(len(compet_ds)):
            class_sign = dict()

            for est in estimators.keys():
                if prediction[est][idx] not in class_sign:
                    class_sign[prediction[est][idx]] = [z_errors[est][idx]]
                else:
                    class_sign[prediction[est][idx]].append(z_errors[est][idx])

            for key in class_sign:
                class_sign[key] = np.mean(class_sign[key])

            selected_classes.append(min(class_sign, key=class_sign.get))
    else:
        for idx in range(len(compet_ds)):
            #class_sign = dict.fromkeys(set(prediction[est].tolist()), 0)
            priority_class = None
            min_error = 1
            for est in estimators.keys():
                if z_errors[est][idx] < min_error:
                    priority_class = prediction[est][idx]
                    min_error = z_errors[est][idx]
            selected_classes.append(priority_class)


    results = accuracy_score(selected_classes, compet_ds[Y_value].tolist())
    return results
    # Predicted the Errors of the models by Random Forest Regression
    
def init_est():
    
    estimators = dict()
        
    estimators['mlr'] = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000000)

    estimators['svm'] = svm.SVC(probability=True)

    estimators['sgd'] = SGDClassifier(loss='log')

    estimators['rfc'] = RandomForestClassifier()

    estimators['multNB'] = MultinomialNB() 

    estimators['bernNB'] = BernoulliNB() 

    leaf_size = list(range(10,50))
    n_neighbors = list(range(1,20))
    p=[1,2]
    hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
    estimators['knn'] = KNeighborsClassifier()
    estimators['knn'] = GridSearchCV(estimators['knn'], hyperparameters, cv=10)

    estimators['ada'] = AdaBoostClassifier()

    return estimators


def analyze_assembly(learning_ds, competition_ds, X, Y, trained_est = None, priority=None, beta=None):
    """
    Innitiate the Ensembling
    """
    if not trained_est:
        est = init_est()
    else:
        est = trained_est
    
    print("Beta " + str(beta))
    
    
    Z_class, predictors = Z_scoring(data=learning_ds, X_values=X, Y_value=Y, beta=beta, estimators = est)

    Z_result = compete(compet_ds=competition_ds, Z_model=Z_class,
                       estimators=predictors, X_values=X, Y_value=Y, priority=priority)
    return Z_result
