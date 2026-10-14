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
        if self.distance == "euclidean":
            batch_sq = np.sum(X_batch ** 2, axis=1, keepdims=True)
            train_sq = np.sum(self.X_train ** 2, axis=1)[None, :]
            squared = batch_sq + train_sq - 2.0 * (X_batch @ self.X_train.T)
            np.maximum(squared, 0.0, out=squared)
            return np.sqrt(squared)

        # dist manhattan acumulada por atributo p poupar memoria
        distances = np.zeros((len(X_batch), len(self.X_train)), dtype=float)
        for j in range(self.X_train.shape[1]):
            distances += np.abs(X_batch[:, j, None] - self.X_train[None, :, j])
        return distances

    def predict(self, X_test):
        X_test = np.asarray(X_test, dtype=float)
        predictions = []
        k = min(self.k, len(self.X_train))

        for start in range(0, len(X_test), KNN_BATCH_SIZE):
            batch = X_test[start:start + KNN_BATCH_SIZE]
            distances = self._distances(batch)
            nearest_indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]

            for row_index, neighbors in enumerate(nearest_indices):
                labels = self.y_train[neighbors]
                if self.task == "classification":
                    values, counts = np.unique(labels, return_counts=True)
                    predictions.append(values[np.argmax(counts)])
                else:
                    predictions.append(float(np.mean(labels)))

        dtype = int if self.task == "classification" else float
        return np.asarray(predictions, dtype=dtype)


class MultipleLinearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        design = np.column_stack((np.ones(len(X_train)), X_train))
        # pinv p evitar problema de matriz singular
        self.coefficients = np.linalg.pinv(design) @ y_train
        return self

    def predict(self, X_test):
        X_test = np.asarray(X_test, dtype=float)
        design = np.column_stack((np.ones(len(X_test)), X_test))
        return design @ self.coefficients


class NaiveBayes:
    def __init__(self, variance_smoothing=1e-6):
        self.variance_smoothing = variance_smoothing
        self._classes = None
        self._mean = None
        self._var = None
        self._prior = None

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train)
        n_samples, n_features = X_train.shape

        self._classes = np.unique(y_train)
        n_classes = len(self._classes)
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._prior = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X_train[y_train == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._prior[idx] = X_c.shape[0] / float(n_samples)

        global_var = np.var(X_train, axis=0)
        floor = self.variance_smoothing * np.maximum(global_var, 1.0)
        self._var = np.maximum(self._var, floor)
        return self

    def predict(self, X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        log_posteriors = []

        for idx, _ in enumerate(self._classes):
            variance = np.maximum(self._var[idx], EPSILON)
            difference = X_test - self._mean[idx]

            log_density = -0.5 * np.sum(
                np.log(2.0 * np.pi * variance) + (difference ** 2) / variance,
                axis=1,
            )
            log_prior = np.log(max(self._prior[idx], EPSILON))
            log_posteriors.append(log_prior + log_density)

        best_indices = np.argmax(np.column_stack(log_posteriors), axis=1)
        return self._classes[best_indices]


class GaussianMultivariateBayes:
    def __init__(self, regularization=1e-4):
        self.regularization = regularization

    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.log_priors = {}
        self.means = {}
        self.inv_covariances = {}
        self.log_determinants = {}
        n_features = X_train.shape[1]
        identity = np.eye(n_features)

        for cls in self.classes:
            X_cls = X_train[y_train == cls]
            self.log_priors[cls] = np.log(len(X_cls) / len(y_train))
            mean = np.mean(X_cls, axis=0)
            self.means[cls] = mean

            if len(X_cls) > 1:
                covariance = np.cov(X_cls, rowvar=False)
            else:
                covariance = identity.copy()
            covariance = np.atleast_2d(covariance)
            
            scale = max(float(np.trace(covariance) / max(n_features, 1)), 1.0)
            covariance = covariance + identity * self.regularization * scale

            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0 or not np.isfinite(logdet):
                covariance = covariance + identity * (self.regularization * 10.0 + EPSILON)
                sign, logdet = np.linalg.slogdet(covariance)

            self.inv_covariances[cls] = np.linalg.pinv(covariance)
            self.log_determinants[cls] = float(logdet)
        return self

    def predict(self, X_test):
        log_probabilities = []
        n_features = X_test.shape[1]

        for cls in self.classes:
            diff = X_test - self.means[cls]
            mahalanobis = np.einsum("bi,ij,bj->b", diff, self.inv_covariances[cls], diff)
            log_density = -0.5 * (
                n_features * np.log(2.0 * np.pi)
                + self.log_determinants[cls]
                + mahalanobis
            )
            log_probabilities.append(self.log_priors[cls] + log_density)

        return self.classes[np.argmax(np.column_stack(log_probabilities), axis=1)]


# main loops
def summarize_results(model_name, resultados_fold, task):
    keys = [key for key in resultados_fold[0] if key != "Matriz de confusão"]
    resumo = {"Algoritmo": model_name}

    for key in keys:
        values = np.asarray([r[key] for r in resultados_fold], dtype=float)
        resumo[key] = (float(np.nanmean(values)), float(np.nanstd(values, ddof=1)))

    train_times = np.asarray([r["Tempo treino (s)"] for r in resultados_fold])
    test_times = np.asarray([r["Tempo teste (s)"] for r in resultados_fold])
    resumo["Tempo treino (s)"] = (float(np.mean(train_times)), float(np.std(train_times, ddof=1)))
    resumo["Tempo teste (s)"] = (float(np.mean(test_times)), float(np.std(test_times, ddof=1)))
    resumo["Matriz de confusão"] = resultados_fold[-1].get("Matriz de confusão")
    return resumo


def evaluate_model_cv(model_name, model_factory, X, y, task, n_folds=N_FOLDS, seed=RANDOM_SEED):
    resultados_fold = []

    for fold_number, (train_idx, test_idx) in enumerate(
        cross_validation_splits(len(X), n_folds=n_folds, seed=seed), start=1
    ):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_scaled, X_test_scaled = standardize_train_test(X_train, X_test)
        
        # print("iniciando fit do fold", fold_number)

        model = model_factory()
        start = time.perf_counter()
        model.fit(X_train_scaled, y_train)
        train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        y_pred = model.predict(X_test_scaled)
        test_seconds = time.perf_counter() - start

        if task == "classification":
            metrics = classification_metrics(y_test, y_pred, positive_label=1)
        else:
            metrics = regression_metrics(y_test, y_pred, n_features=X_train.shape[1])

        metrics["Tempo treino (s)"] = train_seconds
        metrics["Tempo teste (s)"] = test_seconds
        resultados_fold.append(metrics)

        if task == "classification":
            print(f"  Fold {fold_number}: acc={metrics['Acurácia']:.4f}, f1={metrics['F1-score']:.4f}")
        else:
            print(f"  Fold {fold_number}: r2={metrics['R²']:.4f}, mae={metrics['MAE']:.2f}")

    return summarize_results(model_name, resultados_fold, task)


def print_summary_table(resumos, task):
    print("\n--- Resultados ---")
    for resumo in resumos:
        print(f"\nModelo: {resumo['Algoritmo']}")
        for k, v in resumo.items():
            if k not in ["Algoritmo", "Matriz de confusão"]:
                print(f"  {k}: {v[0]:.4f} (+- {v[1]:.4f})")


def save_summary_csv(resumos, task, output_path):
    if task == "classification":
        columns = ["Acurácia", "Precisão", "Recall", "F1-score", "Tempo treino (s)", "Tempo teste (s)"]
    else:
        columns = ["R²", "R² ajustado", "MAE", "RMSE", "Tempo treino (s)", "Tempo teste (s)"]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["Algoritmo"] + [f"{c} media" for c in columns] + [f"{c} std" for c in columns])
        for resumo in resumos:
            means = [resumo[c][0] for c in columns]
            stds = [resumo[c][1] for c in columns]
            writer.writerow([resumo["Algoritmo"]] + means + stds)


def run_classification_experiment():
    print("\nExp 1: Bank Marketing")
    if not CLASSIFICATION_FILE.exists():
        print("arquivo n encontrado")
        return

    X, y, target, atributos, task, target_classes, total_attributes, original_rows = encode_dataset(
        CLASSIFICATION_FILE, MAX_CLASSIFICATION_ROWS
    )
    # print("dimensoes X:", X.shape)
    
    models = [
        ("kNN (Euclidiana)", lambda: KNN(K_NEIGHBORS, "classification", "euclidean")),
        ("kNN (Manhattan)", lambda: KNN(K_NEIGHBORS, "classification", "manhattan")),
        ("NB Univariado", lambda: NaiveBayes()),
        ("NB Multivariado", lambda: GaussianMultivariateBayes()),
    ]

    resumos = []
    for name, factory in models:
        print(f"\nTreinando {name}...")
        resumos.append(evaluate_model_cv(name, factory, X, y, "classification"))

    print_summary_table(resumos, "classification")
    save_summary_csv(resumos, "classification", BASE_DIR / "resultados_classificacao.csv")


def run_regression_experiment():
    print("\nExp 2: Brazilian Houses")
    if not REGRESSION_FILE.exists():
        print("arquivo n encontrado")
        return

    X, y, target, atributos, task, target_classes, total_attributes, original_rows = encode_dataset(
        REGRESSION_FILE,
        MAX_REGRESSION_ROWS,
        drop_features={"hoa_(BRL)", "rent_amount_(BRL)", "property_tax_(BRL)", "fire_insurance_(BRL)"},
    )

    models = [
        ("kNN Reg (Euclidiana)", lambda: KNN(K_NEIGHBORS, "regression", "euclidean")),
        ("kNN Reg (Manhattan)", lambda: KNN(K_NEIGHBORS, "regression", "manhattan")),
        ("Reg Linear", lambda: MultipleLinearRegression()),
    ]

    resumos = []
    for name, factory in models:
        print(f"\nTreinando {name}...")
        resumos.append(evaluate_model_cv(name, factory, X, y, "regression"))

    print_summary_table(resumos, "regression")
    save_summary_csv(resumos, "regression", BASE_DIR / "resultados_regressao.csv")


if __name__ == "__main__":
    run_classification_experiment()
    run_regression_experiment()