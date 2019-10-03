import function
import pandas as pd
import numpy as np

#Method can be 'random' or 'NCR' standing for neighborhood cleaning rule

def master(df,method=None):

    X_train, X_test, y_train, y_test = function.getdataset(df)

    if (method=='random'):
        X_train,y_train = function.random_under_sampling(X_train,y_train)
    elif(method=='NCR'):
        X_train,y_train = function.neighbourhood_clear_rule(X_train, y_train)
    #elif(method=='kmean'):
        #X_train,y_train = function.neighbourhood_clear_rule(X_train, y_train)
    else:
        pass
    
    y_pred_E = function.elasticNet(X_train, y_train, X_test)
    y_pred_RF = function.random_forest(X_train, y_train, X_test)
    y_pred_RF = function.xgboost_model(X_train, y_train, X_test)

    metrics_E = function.compute_metrics(y_test, y_pred_E)
    metrics_RF = function.compute_metrics(y_test, y_pred_RF)
    metrics_XgBoost = function.compute_metrics(y_test, y_pred_RF)

    final_df = pd.DataFrame(np.array(metrics_E,metrics_RF,metrics_XgBoost),
                            columns=['AUC', 'F1-Score', 'MCC', 'Precision of the weak class', 'Recall of the weak class'],
                            index=['Elastic Net', 'Random Forest', 'Xgboost'])
    return final_df

