import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

#Création de la variable cible


def prepare_features(df, features, target, date_col):
    df = df.sort_values(date_col).copy()
    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target]
    return X, y


def temporal_split(X, y, train_size=0.8):
    split = int(len(X) * train_size)
    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]
    return X_train, X_test, y_train, y_test