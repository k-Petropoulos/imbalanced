import imblearn
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import roc_curve, auc, f1_score, matthews_corrcoef
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


#Load data
def getdataset(df):
    X = df.iloc[:,:-1]
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return (X_train, X_test, y_train, y_test)

#Computing and showing metrics
def compute_metrics(y_test, y_pred):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(y_test, y_pred, average='macro')  
    MCC = matthews_corrcoef(y_test, y_pred)  
    return(fpr, tpr, roc_auc, f1, MCC)


def show_AUC(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    return(plt.show())

#Under sampling
def random_under_sampling(X_train, y_train):
    rus = RandomUnderSampler( return_indices=False,random_state=42)
    X_res,y_res= rus.fit_resample(X_train, y_train)
    return X_res,y_res 

def neighbourhood_clear_rule(X_train, y_train):
    ncr = NeighbourhoodCleaningRule()
    X_res, y_res = ncr.fit_resample(X_train, y_train)
    return X_res, y_res

#Algorithm prediction
def random_forestGrid(X_train, y_train,X_test):
    RF = RandomForestClassifier(n_estimators='warn',
                        criterion='gini',
                            max_depth=None,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            min_weight_fraction_leaf=0.0,
                            max_features='auto',
                            max_leaf_nodes=None,
                            min_impurity_decrease=0.0,
                            min_impurity_split=None,
                            bootstrap=True,
                            oob_score=False,
                            n_jobs=None,
                            random_state=None,
                            verbose=0,
                            warm_start=False,
                            class_weight=None)
    # param_grid = { 
    #     'bootstrap': [True, False],
    #     'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    #     'max_features': ['auto', 'sqrt'],
    #     'min_samples_leaf': [1, 2, 4],
    #     'min_samples_split': [2, 5, 10],
    #     'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    # }
    #CV_rf = GridSearchCV(estimator=RF, param_grid=param_grid, cv= 5)
    #CV_rf.fit(X_train, y_train)
    #y_pred = CV_rf.predict(X_test)
    RF.fit()
    RF.fit(X_train, y_train)
    y_pred = RF.predict(X_test)
    return(y_pred)

def elasticNet (X_train, y_train, X_test):
    elasticnet = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                          eps=0.001,
                          n_alphas=100,
                          alphas=None,
                          fit_intercept=True,
                          normalize=False,
                          precompute='auto',
                          max_iter=1000,
                          tol=0.0001,
                          cv=4,
                          copy_X=True,
                          verbose=0,
                          n_jobs=None,
                          positive=False,
                          random_state=None,
                          selection='cyclic')
    elasticnet.fit(X_train,y_train)
    y_pred = elasticnet.predict((X_test))
    return y_pred   

def xgboost_model(X_train, y__train,X_test):
    model = XGBClassifier(max_depth=3, 
                          learning_rate=0.1, 
                          n_estimators=100, 
                          verbosity=1, 
                          silent=None, 
                          objective='binary:logistic', 
                          booster='gbtree', 
                          n_jobs=1, 
                          nthread=None, 
                          gamma=0, 
                          min_child_weight=1, 
                          max_delta_step=0, 
                          subsample=1, 
                          colsample_bytree=1, 
                          colsample_bylevel=1, 
                          colsample_bynode=1, 
                          reg_alpha=0, 
                          reg_lambda=1, 
                          scale_pos_weight=1, 
                          base_score=0.5, 
                          random_state=0, 
                          seed=None, 
                          missing=None)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred