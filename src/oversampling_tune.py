import function
import numpy as np


def tune_oversampling( df, numStrategies=6 ):
    '''
        Split the data, then calculate how many different factors will
        be used, including original data and equal data in both classes.
    '''
    
    # Split data
    X_train, X_test, y_train, y_test = function.getdataset(df)
    
    # Shrink_factors
    Nmin = y_train.value_counts()[1] # number of observations in minority class
    Nmaj = y_train.value_counts()[0] # #number of observations in majorit class
    factor = np.linspace(1.1, Nmaj/Nmin, numStrategies) # factors to expand minority class
    strategy = (Nmin/Nmaj)*factor 
    
    
    average_precision = average_precision_score(y_test, y_pred)
    
    
    # iterate over methods/ models and plot avg precision
    methods = [function.smote_simple, function.smote_borderline, function.adasyn_method]
    models = [function.random_forest, function.xgboost_model]
    for method in methods:
        for model in models:
            avg = []
            for ratio in strategy: 
                X_res, y_res = method( X_train, y_train, strategy= ratio )
                y_pred = model(X_res, y_res, X_test.values)
                avg.append( average_precision_score(y_test, y_pred) )
            plt.plot(strategy, avg)
            plt.xlabel('Over-sampling ratio')
            plt.ylabel('Average Precision')
            plt.title(model)
            plt.show()