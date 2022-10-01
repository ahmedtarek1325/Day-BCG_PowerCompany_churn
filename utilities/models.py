from tkinter import N
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV



from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as XGB



import numpy as np


def logistic_regression(X_train,y_train):
    '''
    INPUT 

    OUTPUT 
    - return fitted logistic regression
    - returns the grid search results
    ACTIONS 
    1. does grid search on the specified parameters 
    2. fits a logistic regression with the best estimator
    '''
    param_grid = {'penalty': ['l1', 'l2'],
                'C': np.arange(0.5, 5, 0.5).tolist()}

    log_model = LogisticRegression(solver = 'liblinear', max_iter=150)
    grid = GridSearchCV(log_model, param_grid, cv=5, scoring='f1',n_jobs=-1)
    grid_search=grid.fit(X_train, y_train)

    print(grid_search.best_params_)
    print('Best Acc = ', grid_search.best_score_*100)
    
    estimator = grid_search.best_params_
    log_model = LogisticRegression(**estimator,solver = 'liblinear', max_iter=150)
    log_model.fit(X_train, y_train)

    
    return log_model, grid_search



def random_forest_random_search(X_train,y_train):
    random_grid = {"criterion":['gini'],
                   'max_depth':np.arange(10,100,10),
                   'min_samples_split':np.arange(15,50,5),
                   'min_samples_leaf':np.arange(10,50,5) }


    RF = RandomForestClassifier()
    rand = RandomizedSearchCV(RF, random_grid, cv=5, scoring='f1',n_jobs=-1)
    random_search=rand.fit(X_train, y_train)

    print(random_search.best_params_)
    print('Best Acc = ', random_search.best_score_*100)
    return random_search
def random_forest(X_train,y_train,hp):
    

    RF = RandomForestClassifier()
    grid = GridSearchCV(RF, hp, cv=5, scoring='f1',n_jobs=-1)
    grid_search=grid.fit(X_train, y_train)

    print(grid_search.best_params_)
    print('Best Acc = ', grid_search.best_score_*100)

    estimator=  grid_search.best_params_
    RF = RandomForestClassifier(**estimator)
    RF.fit(X_train,y_train)


    return RF, grid_search

def SVC_random_search(X_train,y_train):
    hp= {'C': np.logspace(-2, 2, 5),
    'gamma': np.logspace(-2, 2, 5),
    'kernel': ['linear','rbf','poly']}

    rand = RandomizedSearchCV(SVC(), hp, cv=3, scoring='f1',n_jobs=-1)
    random_search=rand.fit(X_train, y_train)

    print(random_search.best_params_)
    print('Best Acc = ', random_search.best_score_*100)

    estimator=  random_search.best_params_
    svc = SVC(**estimator)
    svc.fit(X_train,y_train)


    return svc, random_search
def SVC_grid(X_train,y_train,hp):
    
    
    grid = GridSearchCV(SVC(), hp, cv=3, scoring='f1',n_jobs=-1)
    grid_search=grid.fit(X_train, y_train)

    print(grid_search.best_params_)
    print('Best Acc = ', grid_search.best_score_*100)

    estimator=  grid_search.best_params_
    svc = SVC(**estimator)
    svc.fit(X_train,y_train)


    return svc, grid_search


def XGB_random_Search(X_train, y_train):
    params={ 'max_depth': [4,5,6,7,8] ,
             'learning_rate':[0.05, 0.10, 0.15, 0.20 ] ,
             'gamma':[ 0.0, 0.1, 0.2 , 0.3]}
    xgb= XGB.XGBClassifier(random_state=42)
    random_search = RandomizedSearchCV(xgb,params,scoring='f1',n_iter=25)
    random_search.fit(X_train, y_train)

    print(random_search.best_params_)
    print('Best Acc = ', random_search.best_score_*100)
    

    estimator=  random_search.best_params_
    xgb = XGB.XGBClassifier(**estimator)
    xgb.fit(X_train,y_train)
    return xgb,random_search
