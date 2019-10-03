import subprocess
import sys

# Function to install non-existing modules
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

# packages to check if available at host
pkgs = ['imblearn', 'xgboost']
for package in pkgs:
    try:
        import package
    except ImportError:
        install( package )

import imblearn
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV
<<<<<<< HEAD
from sklearn.metrics import roc_curve, auc, f1_score, matthews_corrcoef, average_precision_score, precision_score, recall_score
=======
from sklearn.metrics import roc_curve, auc, f1_score, matthews_corrcoef, average_precision_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
>>>>>>> c2deb8ded8e777df9f7d5e6d706d99015b63e27e
from imblearn.under_sampling import RandomUnderSampler, NeighbourhoodCleaningRule, CondensedNearestNeighbour, ClusterCentroids
from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE, SVMSMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
<<<<<<< HEAD
from sklearn.cluster import KMeans
=======
from inspect import signature

>>>>>>> c2deb8ded8e777df9f7d5e6d706d99015b63e27e


#Load data
def getdataset(df):
    X = df.iloc[:,:-1]
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return (X_train, X_test, y_train, y_test)


# ===================================== Visualization =====================================

#Computing and showing metrics
def compute_metrics(y_test, y_pred):
    #fpr = dict()
    #tpr = dict()
    #roc_auc = dict()
    #fpr, tpr, _ = roc_curve(y_test, y_pred)
    #roc_auc = auc(fpr, tpr)
    f1 = f1_score(y_test, y_pred, average='macro')  
    MCC = matthews_corrcoef(y_test, y_pred) 
    precisionWeakClass = precision_score(y_test, y_pred, pos_label=1, average='binary')
    recallWeakClass = recall_score(y_test, y_pred, pos_label=1, average='binary')
    confMatrix = plot_confusion_matrix( y_test, y_pred, classes= ['not-fraudulent', 'fraudulent'])
    areaPR = areaUnderPR(y_test, y_pred)
    return([f1, MCC, precisionWeakClass, recallWeakClass, confMatrix, areaPR])


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function computes and plots the confusion matrix.
    Normalization to display percentages can be applied by
    setting `normalize=True`. Based on sklearn example.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    fig, ax = plt.subplots(figsize=(15,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



def areaUnderPR(y_test, y_pred ):
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

# ===================================== Data modification =====================================
#Under sampling
def random_under_sampling(X_train, y_train):
    rus = RandomUnderSampler( return_indices=False,random_state=42)
    X_res,y_res= rus.fit_resample(X_train, y_train)
    return X_res,y_res

def neighbourhood_clear_rule(X_train, y_train):
    ncr = NeighbourhoodCleaningRule()
    X_res, y_res = ncr.fit_resample(X_train, y_train)
    return X_res, y_res
    
def nearest_neighbours(X_train, y_train):
    cnn = CondensedNearestNeighbour(random_state=42)
    X_res, y_res = cnn.fit_resample(X_train, y_train)
    return X_res, y_res

def KMeansUnderSample( X_train, y_train , shrink_factor ): 
    '''
        Creates new majority class by clustering the existing. 
        The class is shrunk according to the "shrink_factor" parameter.
        Under-sample only the majority class and substitute values 
        with the cluster centorids. Returns X, y after subsampling.
    '''
    # check type to be able to use VALUE_COUNTS
    if type(y_train) != pd.core.series.Series: 
        y_train = pd.Series( y_train )
    # minority class count
    Nmin = y_train.value_counts()[1]
    # majority class count (to be created)
    NmajR = y_train.value_counts()[0]/ shrink_factor
    strategy = Nmin/NmajR
    
    cc = ClusterCentroids(random_state= 1, sampling_strategy= strategy, voting= 'soft', estimator= KMeans())
    return cc.fit_sample(X_train, y_train)

#Over sampling
def random_over_sampling(X_train, y_train): RandomOverSampler(random_state= 1).fit_resample(X_train, y_train)

def smote_border(X_train, y_train):
    smote = BorderlineSMOTE( sampling_strategy='not majority',
                             random_state= 1,
                             m_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res

def smote_svm(X_train, y_train):
    smote = SVMSMOTE( sampling_strategy='not majority',
                      random_state= 1,
                      m_neighbors=5,
                      svm_estimator= SVC(kernel= 'linear', gamma= 'scale'))
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res

# ===================================== Models =====================================
#Prediction algorithm

def random_forest(X_train, y_train, X_test):
    RF = RandomForestClassifier(n_estimators=50,
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

    RF.fit(X_train, y_train)
    y_pred = RF.predict(X_test)
    return(y_pred)

def elasticNet(X_train, y_train, X_test):
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
                          n_jobs=-1,
                          positive=False,
                          random_state=None,
                          selection='cyclic')
    elasticnet.fit(X_train,y_train)
    y_pred = elasticnet.predict((X_test))

    # for y in y_pred:
    #     if y <= 0.5:
    #         y = 0
    #     else:
    #         y = 1
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
                          scale_pos_weight=1, 
                          base_score=0.5, 
                          random_state=0, 
                          seed=None, 
                          missing=None)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred
<<<<<<< HEAD

# def KMeansUnderSample(X_train, y_train , shrink ): 
#     '''
#         Creates new majority class by clustering the existing. 
#         The class is shrunk according to the "shrink" parameter.
# 	Returns X, y after subsampling.
#     '''
#     if type(y_train) != pd.core.series.Series: # type check to be abe to us VALUE_COUNTS
#         y_train = pd.Series( y_train)

#     Nmin = y_train.value_counts()[1] # minority class count
#     NmajR = y_train.value_counts()[0]/ shrink # NEW majority class count
#     strategy = Nmin/NmajR
#     # under-sample only the majority class. substitute with the centorids
#     cc = ClusterCentroids(random_state= 1, sampling_strategy= strategy, voting= 'soft', estimator= KMeans())
#     return cc.fit_sample(X_train, y_train)
=======
>>>>>>> c2deb8ded8e777df9f7d5e6d706d99015b63e27e
