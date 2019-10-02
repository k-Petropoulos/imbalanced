import imblearn
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib as plt

def getdataset(df):
    X = df.iloc[:,:-1]
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return (X_train, X_test, y_train, y_test)

