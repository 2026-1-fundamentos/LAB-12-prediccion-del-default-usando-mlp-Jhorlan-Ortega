# flake8: noqa: E501
import gzip
import json
import os
import pickle
import zipfile

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==============================================================================
# Paso 1: Carga y limpieza de datos
# ==============================================================================

def load_data(path: str) -> pd.DataFrame:
    """Lee un CSV comprimido en ZIP y lo retorna como DataFrame."""
    with zipfile.ZipFile(path) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            return pd.read_csv(f)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset:
      - Renombra la columna objetivo.
      - Elimina la columna ID.
      - Elimina filas con información no disponible (valor 0 en
        las columnas categóricas EDUCATION y MARRIAGE).
      - Agrupa EDUCATION > 4 en la categoría 4 (others).
    """
    # Renombrar columna objetivo
    df = df.rename(columns={"default payment next month": "default"})

    # Eliminar columna ID
    df = df.drop(columns=["ID"])

    # Eliminar registros con información no disponible
    df = df[df["EDUCATION"] != 0]
    df = df[df["MARRIAGE"] != 0]

    # Agrupar EDUCATION > 4 → 4 (others)
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

    return df.reset_index(drop=True)


train_df = load_data("files/input/train_data.csv.zip")
test_df  = load_data("files/input/test_data.csv.zip")

train_df = clean_data(train_df)
test_df  = clean_data(test_df)


# ==============================================================================
# Paso 2: Separar features y target
# ==============================================================================

x_train = train_df.drop(columns=["default"])
y_train = train_df["default"]

x_test = test_df.drop(columns=["default"])
y_test = test_df["default"]


# ==============================================================================
# Paso 3: Construcción del pipeline
# ==============================================================================

# Variables categóricas que necesitan OHE; el resto pasa como numérico
CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]

# ColumnTransformer: OHE en categóricas + passthrough en numéricas
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            CATEGORICAL_FEATURES,
        ),
    ],
    remainder="passthrough",  # deja pasar las columnas numéricas sin cambio
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),          # OHE → matriz densa
        ("pca", PCA()),                           # todas las componentes
        ("scaler", StandardScaler()),             # escala a media 0, std 1
        ("selector", SelectKBest(f_classif)),     # K mejores columnas
        ("classifier", MLPClassifier(
            max_iter=2000,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=20,
        )),
    ]
)


# ==============================================================================
# Paso 4: Optimización de hiperparámetros con validación cruzada (10 splits)
# ==============================================================================

param_grid = {
    "selector__k": [20, 30],
    "classifier__hidden_layer_sizes": [(100,), (100, 50)],
    "classifier__alpha": [0.0001, 0.001],
    "classifier__learning_rate_init": [0.001, 0.01],
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="balanced_accuracy",   # métrica requerida
    n_jobs=-1,
    refit=True,
    verbose=1,
)

grid_search.fit(x_train, y_train)

print(f"\n→ Mejores hiperparámetros : {grid_search.best_params_}")
print(f"→ Mejor balanced_accuracy (CV): {grid_search.best_score_:.4f}")


# ==============================================================================
# Paso 5: Guardar el modelo comprimido con gzip
# ==============================================================================

os.makedirs("files/models", exist_ok=True)

with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(grid_search, f)

print("\n✓ Modelo guardado en files/models/model.pkl.gz")


# ==============================================================================
# Pasos 6 y 7: Métricas y matrices de confusión → files/output/metrics.json
# ==============================================================================

os.makedirs("files/output", exist_ok=True)


def compute_metrics_and_cm(model, x, y, dataset_name: str):
    """
    Devuelve dos diccionarios:
      1. Métricas de clasificación (precision, balanced_accuracy, recall, f1).
      2. Matriz de confusión en el formato requerido.
    """
    y_pred = model.predict(x)

    metrics_entry = {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, zero_division=0)),
    }

    cm = confusion_matrix(y, y_pred)
    cm_entry = {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {
            "predicted_0": int(cm[0, 0]),
            "predicted_1": int(cm[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm[1, 0]),
            "predicted_1": int(cm[1, 1]),
        },
    }

    return metrics_entry, cm_entry


train_metrics, train_cm = compute_metrics_and_cm(grid_search, x_train, y_train, "train")
test_metrics,  test_cm  = compute_metrics_and_cm(grid_search, x_test,  y_test,  "test")

# Orden de escritura: métricas train, métricas test, CM train, CM test
with open("files/output/metrics.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(train_metrics) + "\n")
    f.write(json.dumps(test_metrics)  + "\n")
    f.write(json.dumps(train_cm)      + "\n")
    f.write(json.dumps(test_cm)       + "\n")

print("\n✓ Métricas guardadas en files/output/metrics.json")
print(f"\nTrain → {train_metrics}")
print(f"Test  → {test_metrics}")