import pandas as pd
import numpy as np

# from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
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
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred)
    }
    return results


def compare_models(cart_eval, rf_eval, log_eval):
    results = []

    results.append({
        "modele": "CART",
        "f1_score": cart_eval["f1_score"],
        "precision": cart_eval["precision"],
        "recall": cart_eval["recall"]
    })

    results.append({
        "modele": "Random Forest",
        "f1_score": rf_eval["f1_score"],
        "precision": rf_eval["precision"],
        "recall": rf_eval["recall"]
    })

    results.append({
        "modele": "Régression logistique",
        "f1_score": log_eval["f1_score"],
        "precision": log_eval["precision"],
        "recall": log_eval["recall"]
    })

    results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)

    return results_df


# Fonction pour calculer les métriques par station
def metrics_par_station(df_test):
    results = []

    for station, group in df_test.groupby("nom_station"):
        y_true = group["y_true"]
        y_pred = group["y_pred"]

        # éviter les erreurs si une classe est absente
        if len(y_true.unique()) < 2:
            continue

        results.append({
            "nom_station": station,
            "f1_score": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "nb_obs": len(group)
        })

    return pd.DataFrame(results).sort_values("f1_score", ascending=False)