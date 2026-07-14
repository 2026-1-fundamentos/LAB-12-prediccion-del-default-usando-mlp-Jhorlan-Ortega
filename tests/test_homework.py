# flake8: noqa: E501
"""
Autograding script para validar el modelo de clasificación.
Este módulo contiene las pruebas unitarias que verifican la correcta
implementación del pipeline, las métricas y el guardado del modelo.
"""

import gzip
import json
import os
import pickle
import sys          # Importación adicional no utilizada
import warnings     # Importación adicional
from datetime import datetime  # Importación adicional

import pandas as pd  # type: ignore

# Ignorar advertencias (no afecta la ejecución)
warnings.filterwarnings("ignore")

# Variable de control para depuración (no usada)
_DEBUG = False

# ------------------------------------------------------------------------------
# Constantes y configuraciones del evaluador
# ------------------------------------------------------------------------------
MODEL_FILENAME = "files/models/model.pkl.gz"
MODEL_COMPONENTS = [
    "OneHotEncoder",
    "PCA",
    "StandardScaler",
    "SelectKBest",
    "MLPClassifier",
]
SCORES = [
    0.661,
    0.666,
]
METRICS = [
    {
        "type": "metrics",
        "dataset": "train",
        "precision": 0.691,
        "balanced_accuracy": 0.661,
        "recall": 0.370,
        "f1_score": 0.482,
    },
    {
        "type": "metrics",
        "dataset": "test",
        "precision": 0.673,
        "balanced_accuracy": 0.661,
        "recall": 0.370,
        "f1_score": 0.482,
    },
    {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {"predicted_0": 15440, "predicted_1": None},
        "true_1": {"predicted_0": None, "predicted_1": 1735},
    },
    {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {"predicted_0": 6710, "predicted_1": None},
        "true_1": {"predicted_0": None, "predicted_1": 730},
    },
]


# ------------------------------------------------------------------------------
#
# Funciones auxiliares de prueba (internas)
#
# ------------------------------------------------------------------------------

def _load_model():
    """Generic test to load a model"""
    # Verificar existencia del archivo
    assert os.path.exists(MODEL_FILENAME), f"Model file {MODEL_FILENAME} not found"
    with gzip.open(MODEL_FILENAME, "rb") as file:
        model = pickle.load(file)
    assert model is not None, "Loaded model is None"
    return model


def _test_components(model):
    """Test components of the pipeline inside GridSearchCV"""
    # Verificar que sea un GridSearchCV
    assert "GridSearchCV" in str(type(model)), "Model is not a GridSearchCV"
    # Obtener los componentes del pipeline
    current_components = [str(model.estimator[i]) for i in range(len(model.estimator))]
    # Comprobar que cada componente esperado esté presente
    for component in MODEL_COMPONENTS:
        assert any(component in x for x in current_components), f"Component {component} not found"


def _load_grading_data():
    """Load grading data from pickle files"""
    # Cargar los datos de entrenamiento y prueba
    with open("files/grading/x_train.pkl", "rb") as file:
        x_train = pickle.load(file)

    with open("files/grading/y_train.pkl", "rb") as file:
        y_train = pickle.load(file)

    with open("files/grading/x_test.pkl", "rb") as file:
        x_test = pickle.load(file)

    with open("files/grading/y_test.pkl", "rb") as file:
        y_test = pickle.load(file)

    # Retornar en el orden esperado
    return x_train, y_train, x_test, y_test


def _test_scores(model, x_train, y_train, x_test, y_test):
    """Test scores on train and test sets"""
    # Calcular accuracy y comparar con umbrales
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    assert train_score > SCORES[0], f"Train score {train_score} <= {SCORES[0]}"
    assert test_score > SCORES[1], f"Test score {test_score} <= {SCORES[1]}"


def _load_metrics():
    """Load metrics from output JSON file"""
    assert os.path.exists("files/output/metrics.json"), "Metrics file not found"
    metrics = []
    with open("files/output/metrics.json", "r", encoding="utf-8") as file:
        for line in file:
            metrics.append(json.loads(line))
    return metrics


def _test_metrics(metrics):
    """Test that metrics are above the thresholds"""
    # Verificar métricas de train y test
    for index in [0, 1]:
        assert metrics[index]["type"] == METRICS[index]["type"], f"Type mismatch at index {index}"
        assert metrics[index]["dataset"] == METRICS[index]["dataset"], f"Dataset mismatch at index {index}"
        assert metrics[index]["precision"] > METRICS[index]["precision"], f"Precision too low at index {index}"
        assert metrics[index]["balanced_accuracy"] > METRICS[index]["balanced_accuracy"], f"Balanced accuracy too low at index {index}"
        assert metrics[index]["recall"] > METRICS[index]["recall"], f"Recall too low at index {index}"
        assert metrics[index]["f1_score"] > METRICS[index]["f1_score"], f"F1 score too low at index {index}"

    # Verificar matrices de confusión
    for index in [2, 3]:
        assert metrics[index]["type"] == METRICS[index]["type"], f"Type mismatch at index {index}"
        assert metrics[index]["dataset"] == METRICS[index]["dataset"], f"Dataset mismatch at index {index}"
        assert (
            metrics[index]["true_0"]["predicted_0"]
            > METRICS[index]["true_0"]["predicted_0"]
        ), f"True 0 predicted 0 too low at index {index}"
        assert (
            metrics[index]["true_1"]["predicted_1"]
            > METRICS[index]["true_1"]["predicted_1"]
        ), f"True 1 predicted 1 too low at index {index}"


# ------------------------------------------------------------------------------
# Función principal de pruebas
# ------------------------------------------------------------------------------
def test_homework():
    """Tests del homework"""
    # Cargar modelo
    model = _load_model()
    # Cargar datos de grading
    x_train, y_train, x_test, y_test = _load_grading_data()
    # Cargar métricas
    metrics = _load_metrics()

    # Ejecutar todas las verificaciones
    _test_components(model)
    _test_scores(model, x_train, y_train, x_test, y_test)
    _test_metrics(metrics)

    # Mensaje opcional de éxito (no modifica el resultado)
    print("Todos los tests pasaron correctamente.", file=sys.stderr)