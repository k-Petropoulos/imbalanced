import subprocess
import sys

# Function to install non-existing modules
# def install(package):
#     subprocess.call([sys.executable, "-m", "pip", "install", package])

# # packages to check if available at host
# pkgs = ['imblearn', 'xgboost']
# for package in pkgs:
#     try:
#         import package
#     except ImportError:
#         install( package )


import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, f1_score, matthews_corrcoef, average_precision_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from xgboost import XGBClassifier

from imblearn.under_sampling import RandomUnderSampler, NeighbourhoodCleaningRule, CondensedNearestNeighbour, ClusterCentroids
from inspect import signature
from imblearn.over_sampling import SMOTE,BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek 


############# Load Data set #############
def getdataset(df):
    X = df.iloc[:,:-1]
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    return (X_train, X_test, y_train, y_test)

############# Metrics #############
    
def compute_metrics(y_test, y_pred):
    f1 = f1_score(y_test, y_pred, average='macro')  
    MCC = matthews_corrcoef(y_test, y_pred) 
    precisionWeakClass = precision_score(y_test, y_pred, pos_label=1, average='binary')
    recallWeakClass = recall_score(y_test, y_pred, pos_label=1, average='binary')
    confMatrix = plot_confusion_matrix( y_test, y_pred, classes= ['not-fraudulent', 'fraudulent'])
    average_precision = average_precision_score(y_test, y_pred)
    return([f1, MCC, precisionWeakClass, recallWeakClass, average_precision])

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    fig, ax = plt.subplots(figsize=(7,5))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[1]),
           xticklabels=classes, yticklabels=classes,
#            title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, s= format(cm[i,j], fmt), va="bottom", ha="center", fontsize=18)
#                     color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def areaUnderPR(y_test, y_pred):
    '''
        Plots the Precision-Recall curve and 
        displays the area under the curve (average precision).
    '''
    average_precision = average_precision_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)

    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    ax = plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format( average_precision) )
    return ax

def compute_AUC (y_test, y_pred):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def show_AUC(fpr, tpr, roc_auc):
    '''
        Plots the AUC and 
        displays the area under the curve (average precision).
    '''
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

############# Under sampling #############

def random_under_sampling(X_train, y_train, strategy='auto'):
    rus = RandomUnderSampler(return_indices=False, sampling_strategy= strategy, random_state= 1)
    X_res,y_res = rus.fit_resample(X_train, y_train)
    return X_res,y_res

def neighbourhood_clear_rule(X_train, y_train, strategy='auto'):
    # Cleaning algorithms do not accept ratio argument
    X_res, y_res = NeighbourhoodCleaningRule(sampling_strategy= 'auto', random_state = 1).fit_resample(X_train, y_train)
    return X_res, y_res
    
def nearest_neighbours(X_train, y_train, strategy='auto'):
    # Cleaning algorithms do not accept ratio argument
    X_res, y_res = CondensedNearestNeighbour(sampling_strategy= 'auto', random_state = 1).fit_resample(X_train, y_train)
    return X_res, y_res

def KMeansUnderSample( X_train, y_train , strategy='auto'): 
    '''
        Creates new majority class by clustering the existing. 
        The class is shrunk according to the "shrink_factor" parameter.
        Under-sample only the majority class and substitute values 
        with the cluster centorids. Returns X, y after subsampling.
    '''
#  #  #  Older #  #  #
#     # check type to be able to use VALUE_COUNTS
#     if type(y_train) != pd.core.series.Series: 
#         y_train = pd.Series( y_train )
#     # minority class count
#     Nmin = y_train.value_counts()[1]
#     # majority class count (to be created)
#     NmajR = y_train.value_counts()[0]/ shrink_factor
#     strategy = Nmin/NmajR
#  #  #  #  #  #  #  #
    cc = ClusterCentroids(random_state= 1, sampling_strategy= strategy, voting= 'soft', estimator= KMeans())
    return cc.fit_sample(X_train, y_train)

############# Over sampling #############

def smote_simple(X_train, y_train, strategy='auto'):
    X_res, y_res = SMOTE(sampling_strategy= strategy, random_state = 1).fit_resample(X_train, y_train)
    return X_res, y_res

def smote_borderline(X_train, y_train, strategy='auto'):
    X_res, y_res = BorderlineSMOTE(sampling_strategy= strategy, random_state = 1).fit_resample(X_train, y_train)
    return X_res, y_res

def adasyn_method(X_train, y_train, strategy='auto'):
    X_res, y_res = ADASYN(sampling_strategy= strategy, random_state = 1).fit_resample(X_train, y_train)
    return X_res, y_res

############# Combined over- and under- sampling #############

def smote_enn(X_train, y_train, strategy):
    X_res, y_res = SMOTEENN(sampling_strategy = strategy, random_state = 1).fit_resample(X_train, y_train)
    return X_res, y_res
    
def smote_tomek(X_train, y_train, strategy):
    smt = SMOTETomek(sampling_strategy = strategy, random_state = 1)
    X_res, y_res = smt.fit_resample(X_train, y_train)    
    return X_res, y_res

############# Fine tune samples #############

def plot_tune_sampling( df, methods, numStrategies=6 ):
    '''
        Split the data, then calculate how many different factors will
        be used, including original data and equal data in both classes.
    '''
    # Handle if not list
    if type(methods) is not list: methods = [ methods ]
    
    # Split data
    X_train, X_test, y_train, y_test = getdataset(df)
    
    # Shrink_factors
    Nmin = y_train.value_counts()[1] # number of observations in minority class
    Nmaj = y_train.value_counts()[0] # #number of observations in majorit class
    factor = np.linspace(1.1, Nmaj/Nmin, numStrategies) # factors to expand minority class
    strategy = (Nmin/Nmaj)*factor 
    
    
    # iterate over methods/ models and plot avg precision
    models = [random_forest, xgboost_model, elasticNet]
    for method in methods:
        for model in models:
            ratios = []
            avg = []
            for ratio in strategy:
                if (method == adasyn_method) and (ratio < 0.0047133): # limit value before throwing ValueError discovered
                    continue
                else:
                    X_res, y_res = method( X_train, y_train, strategy= ratio )
                    y_pred = model(X_res, y_res, X_test.values)
                    avg.append( average_precision_score(y_test, y_pred) )
                    ratios.append( ratio )
                    
            f, ax = plt.subplots(figsize=(8,5))
            ax.plot(ratios, avg)
            plt.xlabel('Over-sampling ratio')
            plt.ylabel('Average Precision')
            method_name = re.search(r"\s\w*", str(method))[0]
            model_name = re.search(r"\s\w*", str(model))[0]
            plt.title( method_name+":  "+model_name  ) # RegEx to capture just what's needed
            yield ax

            
            
def df_tune_sampling( df, methods, numStrategies=10):
    '''
        Similar to the previous, this time instead of yielding a plot
        a DataFrame is constructed with the optimal over/under-sampling size.
    '''
    # Handle if not list
    if type(methods) is not list: methods = [ methods ]
    
    # Split data
    X_train, X_test, y_train, y_test = getdataset(df)
    
    # Shrink_factors
    Nmin = y_train.value_counts()[1] # number of observations in minority class
    Nmaj = y_train.value_counts()[0] # #number of observations in majorit class
    factor = np.linspace(1.1, Nmaj/Nmin, numStrategies) # factors to expand minority class
    strategy = (Nmin/Nmaj)*factor 
    
    
    # iterate over methods/ models and plot avg precision
    models = [random_forest, xgboost_model]
    rows = []
    for method in methods:
        for model in models:
            ratios = []
            avg = []
            for ratio in strategy:
                if (method == adasyn_method) and (ratio < 0.0047133): # limit value before throwing ValueError discovered
                    continue
                else:
                    X_res, y_res = method( X_train, y_train, strategy= ratio )
                    y_pred = model(X_res, y_res, X_test.values)
                    avg.append( average_precision_score(y_test, y_pred) )
                    ratios.append( ratio )
            method_name = re.search(r"\s\w*", str(method))[0]
            model_name = re.search(r"\s\w*", str(model))[0]
            # Find the ratio that maximizes the assessment metric
            t1 = {'method':method_name, 'model':model_name}
            t2 = [{'ratio':ratios[maxInd], 'avg_prcs':maxVal} for maxInd, maxVal in enumerate( avg ) if maxVal == max(avg)]
            rows.append( {**t1, **t2[0]} ) # concatenate in one dict
    return pd.DataFrame( rows )


def tune_OverSampling( X_train, y_train, X_test, y_test, methods, numStrategies=6 ):
    '''
        Similar to the previous, for arrays (already over-sampled)
    '''
    # Handle if not list
    if type(methods) is not list: methods = [ methods ]
    
    if type(y_train) != pd.core.series.Series: # type check to be abe to us VALUE_COUNTS
        y_train = pd.Series( y_train )
    
    # Shrink_factors
    Nmin = y_train.value_counts()[1] # number of observations in minority class
    Nmaj = y_train.value_counts()[0] # #number of observations in majorit class
    factor = np.linspace(1.1, Nmaj/Nmin, numStrategies) # factors to expand minority class
    strategy = (Nmin/Nmaj)*factor 
    
    
    # iterate over methods/ models and plot avg precision
    models = [xgboost_model]
    rows = []
    for method in methods:
        for model in models:
            ratios = []
            avg = []
            for ratio in strategy:
                if (method == adasyn_method) and (ratio < 0.0047133): # limit value before throwing ValueError discovered
                    continue
                else:
                    X_res, y_res = method( X_train, y_train, strategy= ratio )
                    y_pred = model(X_res, y_res, X_test.values)
                    avg.append( average_precision_score(y_test, y_pred) )
                    ratios.append( ratio )
            method_name = re.search(r"\s\w*", str(method))[0]
            model_name = re.search(r"\s\w*", str(model))[0]
            # Find the ratio that maximizes the assessment metric
            t1 = {'method':method_name, 'model':model_name}
            t2 = [{'ratio':ratios[maxInd], 'avg_prcs':maxVal} for maxInd, maxVal in enumerate( avg ) if maxVal == max(avg)]
            rows.append( {**t1, **t2[0]} ) # concatenate in one dict
    return pd.DataFrame( rows )


############# Prediction algorithm #############

def random_forest(X_train, y_train, X_test):
    RF = RandomForestClassifier(n_estimators=100,
                        criterion='gini',
                            max_depth=4,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            min_weight_fraction_leaf=0.0,
                            max_features='auto',
                            max_leaf_nodes=None,
                            min_impurity_decrease=0.0,
                            min_impurity_split=None,
                            bootstrap=True,
                            oob_score=False,
                            n_jobs=-1,
                            random_state=None,
                            verbose=0,
                            warm_start=False)

    RF.fit(X_train, y_train)
    y_pred = RF.predict(X_test)
    return(y_pred)

def elasticNet(X_train, y_train, X_test):
    elasticnet = LogisticRegression(penalty='elasticnet',
                                    multi_class='ovr',
                                    solver='saga',
                                   l1_ratio=0.5,
                                   max_iter= 5000) # raised a lot; would not converge
    elasticnet.fit(X_train,y_train)
    y_pred = elasticnet.predict((X_test))
    return y_pred  

def xgboost_model(X_train, y_train, X_test):
    model = XGBClassifier(max_depth=3, 
                          learning_rate=0.1, 
                          n_estimators=50, 
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
                          base_score=0.5, 
                          random_state=0, 
                          seed=None, 
                          missing=None)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred


############# Algorithm Modifications #############

def grid_search_dict_RF(n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, class_weight):

    # Function to generate a dictionary of the most impactful hyperparameters to be fed into the randomized grid search algorithm  
    # for the Random Forest classifier

    random_grid_RF={'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'class_weight': class_weight
                    }

    return random_grid_RF


def grid_search_dict_XGB(min_child_weight, gamma, subsample, colsample_bytree, max_depth, scale_pos_weight):

    # Function to generate a dictionary of the most impactful hyperparameters to be fed into the randomized grid search algorithm 
    # for the XGBoost classifier

    random_grid_XGB = {'min_child_weight': min_child_weight,
                      'gamma': gamma,
                      'subsample': subsample,
                      'colsample_bytree': colsample_bytree,
                      'max_depth': max_depth,
                      'scale_pos_weight': scale_pos_weight
                      }

    return random_grid_XGB
 

def grid_search_random(X_res, y_res, X_test, method:str, random_grid, n_iter:int, cv:int):

    # Function to perform a randomized hyperparameter grid search as a first guess for the grid of hyperparameter values 

    if (method == "RF"):
        clf = RandomForestClassifier()
    elif (method == "XGB"):
        clf = XGBClassifier()
    else:
        print("Error: specify either 'RF' for Random Forest or 'XGB' for XGBoost")
    
    clf_grid = RandomizedSearchCV(estimator = clf, 
                                  param_distributions = random_grid,   
                                  n_iter = n_iter,
                                  n_jobs= -1,
                                  cv = cv,
                                  verbose = 2,
                                  random_state = 7)

    clf_grid.fit(X_res, y_res)
    best_param = clf_grid.best_params_
    best_model = clf_grid.best_estimator_
    y_pred = best_model.predict(X_test)
    print(best_param)

    return best_param, y_pred


def grid_search_CV(X_res, y_res, X_test, method:str, random_grid, n_iter:int, cv:int):

    # Function to perform a hyperparameter grid search

    if (method == "RF"):
        clf = RandomForestClassifier()
    elif (method == "XGB"):
        clf = XGBClassifier()
    else:
        print("Error: specify either 'RF' for Random Forest or 'XGB' for XGBoost")
    
    clf_grid_cv =  GridSearchCV(estimator = clf, 
                                param_distributions = random_grid,   
                                n_iter = n_iter,
                                n_jobs = -1,
                                cv = cv,
                                verbose = 2)

    clf_grid_cv.fit(X_res, y_res)
    best_param_cv = clf_grid_cv.best_params_
    best_model_cv = clf_grid_cv.best_estimator_
    y_pred = best_model_cv.predict(X_test)
    
    return best_model_cv, y_pred
