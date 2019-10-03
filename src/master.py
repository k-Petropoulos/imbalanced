from src import function


#Method can be 'random' or 'NCR' standing for neighborhood cleaning rule

def master(df,method):

    X_train, X_test, y_train, y_test = function.getdataset(df)
    if (method=='random'):
        X_train,y_train = function.random_under_sampling(X_train,y_train)
    elif(method == 'NCR'):
        X_train,y_train = function.random_under_sampling(X_train, y_train)

    y_pred_E = function.elasticNet(X_train, y_train, X_test)
    y_pred_RF = function.random_forest(X_train, y_train, X_test)
    y_pred_XgBoost = function.xgboost_model(X_train, y_train, X_test)

