from dependencies import *

def load_data(path: str, resp: str, drop: bool = True):
    '''
    Function to load data from specified path. NaN are 
    cleaned according to *drop argument and the data is 
    split according to the response variable *resp.
    '''
    df = pd.read_csv(path)
    if drop:
        df.dropna(inplace= True)
    X = df.drop(resp, axis=1)
    y = df[resp]
    return X, y