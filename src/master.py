import function

#Method can be 'random' or 'NCR' standing for neighborhood cleaning rule, or CNN condensed clear rules or Kmeans

def master(df,method=None,shrink=None):

    X_train, X_test, y_train, y_test = function.getdataset(df)
    
    #under sampling
    if (method=='random'):
        X_train,y_train = function.random_under_sampling(X_train,y_train)
    elif(method=='NCR'):
        X_train,y_train = function.neighbourhood_clear_rule(X_train, y_train)
    elif(method=='CNN'):
        X_train,y_train = function.nearest_neighbours(X_train, y_train)
    elif(method=='ClusterCentroids'):
        X_train,y_train = function.KMeansUnderSample(X_train, y_train, shrink)
    
    #over sampling
    elif(method=='SMOTE'):
        X_train,y_train = function.smote_simple(X_train, y_train)
    elif(method=='SMOTE_border'):
        X_train,y_train = function.smote_borderline(X_train, y_train)
    elif(method=='ADASYN'):
        X_train,y_train = function.adasyn_method(X_train, y_train)

    #If no resampling
    else:
        pass
    
    print('ElasticNet begin')
    y_pred_E = function.elasticNet(X_train, y_train, X_test)
    print('RandomForest begin')
    y_pred_RF = function.random_forest(X_train, y_train, X_test)
    print('Xgboost begin')
    y_pred_XgBoost = function.xgboost_model(X_train, y_train, X_test) # changed X_test-> X_test.values as it threw error

    print('ElasticNet metrics')
    metrics_E = function.compute_metrics(y_test, y_pred_E)
    print('RandomForest metrics')
    metrics_RF = function.compute_metrics(y_test, y_pred_RF)
    print('Xgboost metrics')
    metrics_XgBoost = function.compute_metrics(y_test, y_pred_XgBoost)

    print('Create final dataframe')

#     final_df = pd.DataFrame(np.array(metrics_RF,metrics_XgBoost),
#                             columns=['F1-Score', 'MCC', 'Precision of the weak class', 'Recall of the weak class'])
    return ([metrics_E,metrics_RF,metrics_XgBoost])
