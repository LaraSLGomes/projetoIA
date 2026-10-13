import csv
import math
import time
from pathlib import Path
import numpy as np

# configs globais
BASE_DIR = Path(__file__).resolve().parent
CLASSIFICATION_FILE = BASE_DIR / "phpkIxskf.arff"
REGRESSION_FILE = BASE_DIR / "dataset.arff"

N_FOLDS = 5
RANDOM_SEED = 42
K_NEIGHBORS = 5
MAX_CLASSIFICATION_ROWS = 4000
MAX_REGRESSION_ROWS = 3000
KNN_BATCH_SIZE = 64
EPSILON = 1e-6

# leitura de arquivos arff
def split_arff_csv(line):
    return next(csv.reader([line], delimiter=",", quotechar="'", skipinitialspace=True))

def clean_value(value):
    return str(value).strip().strip("'").strip('"').strip()

def parse_arff(file_path):
    atributos = []
    attribute_types = []
    linhas = []
    in_data = False

    with open(file_path, "r", encoding="utf-8", errors="replace") as arff:
        for raw_line in arff:
            line = raw_line.strip()
            if not line or line.startswith("%") or line.startswith("%%"):
                continue

            lower = line.lower()
            if lower.startswith("@relation"):
                continue

            if lower.startswith("@attribute"):
                parts = line.split(None, 2)
                if len(parts) < 3:
                    continue
                name = clean_value(parts[1])
                definition = parts[2].strip()

                if definition.startswith("{") and definition.endswith("}"):
                    nominal_text = definition[1:-1]
                    nominal_values = [clean_value(v) for v in split_arff_csv(nominal_text)]
                    attribute_types.append(("nominal", nominal_values))
                else:
                    attribute_types.append(("numeric", None))
                atributos.append(name)
                continue

            if lower == "@data":
                in_data = True
                continue

            if in_data:
                try:
                    values = [clean_value(v) for v in split_arff_csv(line)]
                except (csv.Error, StopIteration):
                    continue
                if len(values) == len(atributos):
                    linhas.append(values)

    if not atributos or not linhas:
        raise ValueError(f"Arquivo ARFF vazio ou inválido: {file_path}")

    return atributos, attribute_types, linhas

def encode_dataset(file_path, max_rows=None, seed=RANDOM_SEED, drop_features=None):
    nomes, types, linhas = parse_arff(file_path)
    rng = np.random.default_rng(seed)

    if max_rows is not None and len(linhas) > max_rows:
        selected = rng.choice(len(linhas), size=max_rows, replace=False)
        selected.sort()
        linhas = [linhas[i] for i in selected]

    target_name = nomes[-1]
    target_type, target_nominal = types[-1]
    drop_features = set(drop_features or [])
    feature_indices = [j for j, name in enumerate(nomes[:-1]) if name not in drop_features]
    atributos = [nomes[j] for j in feature_indices]
    feature_types = [types[j] for j in feature_indices]

    raw_X = [[row[j] for j in feature_indices] for row in linhas]
    raw_y = [row[-1] for row in linhas]

    encoded_columns = []
    expanded_atributos = []

    for j, (feature_name, (kind, categories)) in enumerate(zip(atributos, feature_types)):
        column = [clean_value(row[j]) for row in raw_X]
        missing = [value == "?" or value == "" for value in column]

        if kind == "nominal":
            valid = [v for v, miss in zip(column, missing) if not miss]
            if categories:
                category_list = list(categories)
            else:
                category_list = sorted(set(valid))
            if not category_list:
                category_list = ["unknown"]

            if valid:
                values, counts = np.unique(np.asarray(valid, dtype=str), return_counts=True)
                mode_value = values[np.argmax(counts)]
            else:
                mode_value = category_list[0]

            column = [mode_value if miss else v for v, miss in zip(column, missing)]
            for category in category_list:
                encoded_columns.append(np.asarray([1.0 if v == category else 0.0 for v in column]))
                expanded_atributos.append(f"{feature_name}={category}")
        else:
            numeric = []
            for value in column:
                try:
                    numeric.append(float(value) if value != "?" and value != "" else np.nan)
                except ValueError:
                    numeric.append(np.nan)
            numeric = np.asarray(numeric, dtype=float)
            finite = numeric[np.isfinite(numeric)]
            fill_value = float(np.mean(finite)) if len(finite) else 0.0
            numeric[~np.isfinite(numeric)] = fill_value
            encoded_columns.append(numeric)
            expanded_atributos.append(feature_name)

    X = np.column_stack(encoded_columns).astype(float)

    if target_type == "nominal":
        classes = list(target_nominal or sorted(set(raw_y)))
        class_map = {label: index for index, label in enumerate(classes)}
        y = np.asarray([class_map.get(value, 0) for value in raw_y], dtype=int)
        task = "classification"
        target_classes = classes
    else:
        try:
            y = np.asarray([float(value) for value in raw_y], dtype=float)
        except ValueError as exc:
            raise ValueError(f"erro var alvo {target_name}") from exc
        task = "regression"
        target_classes = None

    return X, y, target_name, expanded_atributos, task, target_classes, len(nomes), len(parse_arff(file_path)[2])


# validacao cruzada
def standardize_train_test(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std < EPSILON] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std

def make_folds(n_samples, n_folds=N_FOLDS, seed=RANDOM_SEED):
    if n_samples < n_folds:
        raise ValueError("erro: n_samples menor q folds")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    return [fold for fold in np.array_split(indices, n_folds)]

def cross_validation_splits(n_samples, n_folds=N_FOLDS, seed=RANDOM_SEED):
    folds = make_folds(n_samples, n_folds, seed)
    all_indices = np.arange(n_samples)
    for test_indices in folds:
        mask = np.ones(n_samples, dtype=bool)
        mask[test_indices] = False
        yield all_indices[mask], test_indices

# metricas classificacao
def confusion_matrix_binary(y_true, y_pred, positive_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    negative_label = 0 if positive_label != 0 else 1

    vn = int(np.sum((y_true == negative_label) & (y_pred == negative_label)))
    fp = int(np.sum((y_true == negative_label) & (y_pred == positive_label)))
    fn = int(np.sum((y_true == positive_label) & (y_pred == negative_label)))
    vp = int(np.sum((y_true == positive_label) & (y_pred == positive_label)))
    return np.array([[vn, fp], [fn, vp]], dtype=int)

def classification_metrics(y_true, y_pred, positive_label=1):
    cm = confusion_matrix_binary(y_true, y_pred, positive_label)
    vn, fp = cm[0, 0], cm[0, 1]
    fn, vp = cm[1, 0], cm[1, 1]

    accuracy = (vp + vn) / max(vp + vn + fp + fn, 1)
    precision = vp / max(vp + fp, 1)
    recall = vp / max(vp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, EPSILON)

    return {
        "Acurácia": accuracy,
        "Precisão": precision,
        "Recall": recall,
        "F1-score": f1,
        "Matriz de confusão": cm,
    }

# metricas regressao
def regression_metrics(y_true, y_pred, n_features):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors ** 2))
    rmse = math.sqrt(mse)
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > EPSILON else 0.0

    n = len(y_true)
    if n - n_features - 1 > 0:
        r2_adjusted = 1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1)
    else:
        r2_adjusted = float("nan")

    return {
        "R²": r2,
        "R² ajustado": r2_adjusted,
        "MAE": mae,
        "RMSE": rmse,
    }

# algoritmos
class KNN:
    def __init__(self, k=5, task="classification", distance="euclidean"):
        self.k = k
        self.task = task
        self.distance = distance
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train)
        return self

    def _distances(self, X_batch):
        if self.