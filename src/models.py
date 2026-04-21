import pandas as pd
import numpy as np

# from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def prepare_features(df, features, target):
    df = df.copy()
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


# Le modèle CART
def train_cart(X_train, y_train, max_depth=5, min_samples_leaf=50, random_state=42):
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


# La forêt aléatoire
def train_random_forest(
    X_train,
    y_train,
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=50,
    random_state=42
):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# Le modèle de regression logistique
def train_logistic_regression(X_train, y_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler, X_train_scaled, X_test_scaled


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    results = {
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred)
    }
    return results

