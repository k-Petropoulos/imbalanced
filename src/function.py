import imblearn
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib as plt
from sklearn.linear_model import ElasticNetCV
from imblearn.under_sampling import RandomUnderSampler

def getdataset(df):
    X = df.iloc[:,:-1]
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return (X_train, X_test, y_train, y_test)

def labels(df):
    return list(df)

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
    
def random_under_sampling(X_train, y_train,labels):
    rus = RandomUnderSampler( return_indices=False,random_state=42)
    X_res,y_res= rus.fit_resample(X_train, y_train)
    X_resampled=np.c_[ X_res, y_res ]
    return pd.DataFrame(data=X_resampled, columns=labels)