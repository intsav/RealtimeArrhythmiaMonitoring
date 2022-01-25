import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler

np.random.seed(42)

def load_data():
    path = 'data/'

    df = pd.read_csv(path + "mitdb_360_train.csv", header=None)
    df_test = pd.read_csv(path + "mitdb_360_test.csv", header=None)

    df_val_train = df.values
    X = df_val_train[:, :-1]
    y = df_val_train[:, -1].astype(int)
    
    df_val_test = df_test.values
    X_test = df_val_test[:, :-1]
    Y_test = df_val_test[:, -1].astype(int)

    ros = RandomOverSampler(random_state=0)
    X_train, Y_train = ros.fit_resample(X, y)
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, Y_train, Y_val = train_test_split(
                                        X_train, 
                                        Y_train, 
                                        test_size=0.25, 
                                        random_state=42)
    
    shuffle_idx = np.random.permutation(list(range(X_train.shape[0])))
    X_train = X_train[shuffle_idx]
    Y_train = Y_train[shuffle_idx]
    
    X_train = np.expand_dims(X_train, 2)
    X_val = np.expand_dims(X_val, 2)
    X_test = np.expand_dims(X_test, 2)
    
    ohe = OneHotEncoder()
    Y_test = ohe.fit_transform(Y_test.reshape(-1,1))
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test