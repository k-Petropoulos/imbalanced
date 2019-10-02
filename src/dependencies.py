import subprocess
import sys

def install(package):
    '''
        Used to install non-existing modules
    '''
    subprocess.call([sys.executable, "-m", "pip", "install", package])

pkgs = ['imblearn', 'xgboost']
for package in pkgs:
    try:
        import package
    except ImportError:
        install( package )

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, f1_score, average_precision_score, precision_recall_curve
from inspect import signature