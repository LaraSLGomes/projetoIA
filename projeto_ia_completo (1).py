import csv
import math
import time
from pathlib import Path

import numpy as np  



BASE_DIR = Path(__file__).resolve().parent
CLASSIFICATION_FILE = BASE_DIR / "phpkIxskf.arff"
REGRESSION_FILE = BASE_DIR / "dataset.arff"

N_FOLDS = 5
RANDOM_SEED = 42
K_NEIGHBORS = 5
MAX_CLASSIFICATION_ROWS = None  
MAX_REGRESSION_ROWS = None 
KNN_BATCH_SIZE = 64
EPSILON = 1e-6

def split_arff_csv(line):
    return next(csv.reader([line], delimiter=",", quotechar="'", skipinitialspace=True))


def clean_value(value):
    return str(value).strip().strip("'").strip('"').strip()


def parse_arff(file_path):
    attributes = []
    attribute_types = []
    rows = []
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
                attributes.append(name)
                continue

            if lower == "@data":
                in_data = True
                continue

            if in_data:
                try:
                    values = [clean_value(v) for v in split_arff_csv(line)]
                except (csv.Error, StopIteration):
                    continue
                if len(values) == len(attributes):
                    rows.append(values)

    if not attributes or not rows:
        raise ValueError(f"Arquivo ARFF vazio ou inválido: {file_path}")

    return attributes, attribute_types, rows


def encode_dataset(file_path, max_rows=None, seed=RANDOM_SEED, drop_features=None):
    names, types, rows = parse_arff(file_path)
    rng = np.random.default_rng(seed)

    if max_rows is not None and len(rows) > max_rows:
        selected = rng.choice(len(rows), size=max_rows, replace=False)
        selected.sort()
        rows = [rows[i] for i in selected]

    target_name = names[-1]
    target_type, target_nominal = types[-1]
    drop_features = set(drop_features or [])
    feature_indices = [j for j, name in enumerate(names[:-1]) if name not in drop_features]
    feature_names = [names[j] for j in feature_indices]
    feature_types = [types[j] for j in feature_indices]

    raw_X = [[row[j] for j in feature_indices] for row in rows]
    raw_y = [row[-1] for row in rows]

    encoded_columns = []
    expanded_feature_names = []

    for j, (feature_name, (kind, categories)) in enumerate(zip(feature_names, feature_types)):
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
                expanded_feature_names.append(f"{feature_name}={category}")
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
            expanded_feature_names.append(feature_name)

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
            raise ValueError(f"A variável-alvo {target_name} não é numérica.") from exc
        task = "regression"
        target_classes = None

    return X, y, target_name, expanded_feature_names, task, target_classes, len(names), len(parse_arff(file_path)[2])


# Pré-processamento e validação cruzada

def standardize_train_test(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std < EPSILON] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


def make_folds(n_samples, n_folds=N_FOLDS, seed=RANDOM_SEED):
    if n_samples < n_folds:
        raise ValueError("O número de amostras deve ser maior ou igual ao número de folds.")
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


# Métricas de classificação
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


# Métricas de regressão
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


# kNN — classificação e regressão
class KNN:
    def __init__(self, k=5, task="classification", distance="euclidean"):
        if task not in ("classification", "regression"):
            raise ValueError("task deve ser 'classification' ou 'regression'.")
        if distance not in ("euclidean", "manhattan"):
            raise ValueError("distance deve ser 'euclidean' ou 'manhattan'.")
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


# Regressão Linear Múltipla - solução por pseudoinversa
class MultipleLinearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X_train, y_train):
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        design = np.column_stack((np.ones(len(X_train)), X_train))
        self.coefficients = np.linalg.pinv(design, rcond=1e-12) @ y_train
        return self

    def predict(self, X_test):
        X_test = np.asarray(X_test, dtype=float)
        design = np.column_stack((np.ones(len(X_test)), X_test))
        return design @ self.coefficients


# Classificador Bayesiano Gaussiano univariado (Naive Bayes)
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
                np.log(2.0 * np.pi * variance)
                + (difference ** 2) / variance,
                axis=1,
            )
            log_prior = np.log(max(self._prior[idx], EPSILON))
            log_posteriors.append(log_prior + log_density)

        best_indices = np.argmax(np.column_stack(log_posteriors), axis=1)
        return self._classes[best_indices]


# Classificador Bayesiano Gaussiano multivariado
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
            # Regularização para evitar matriz singular.
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
            mahalanobis = np.einsum(
                "bi,ij,bj->b", diff, self.inv_covariances[cls], diff
            )
            log_density = -0.5 * (
                n_features * np.log(2.0 * np.pi)
                + self.log_determinants[cls]
                + mahalanobis
            )
            log_probabilities.append(self.log_priors[cls] + log_density)

        return self.classes[np.argmax(np.column_stack(log_probabilities), axis=1)]


# Execução e agregação dos experimentos
def summarize_results(model_name, fold_results, task):
    keys = [key for key in fold_results[0] if key != "Matriz de confusão"]
    summary = {"Algoritmo": model_name}

    for key in keys:
        values = np.asarray([result[key] for result in fold_results], dtype=float)
        summary[key] = (float(np.nanmean(values)), float(np.nanstd(values, ddof=1)))

    train_times = np.asarray([r["Tempo treino (s)"] for r in fold_results])
    test_times = np.asarray([r["Tempo teste (s)"] for r in fold_results])
    summary["Tempo treino (s)"] = (float(np.mean(train_times)), float(np.std(train_times, ddof=1)))
    summary["Tempo teste (s)"] = (float(np.mean(test_times)), float(np.std(test_times, ddof=1)))
    summary["Matriz de confusão"] = fold_results[-1].get("Matriz de confusão")
    return summary


def evaluate_model_cv(model_name, model_factory, X, y, task, n_folds=N_FOLDS, seed=RANDOM_SEED):
    fold_results = []

    for fold_number, (train_idx, test_idx) in enumerate(
        cross_validation_splits(len(X), n_folds=n_folds, seed=seed), start=1
    ):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_scaled, X_test_scaled = standardize_train_test(X_train, X_test)

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
        fold_results.append(metrics)

        if task == "classification":
            print(
                f"  Fold {fold_number}/{n_folds}: "
                f"acurácia={metrics['Acurácia']:.4f}, "
                f"F1={metrics['F1-score']:.4f}, "
                f"treino={train_seconds:.3f}s, teste={test_seconds:.3f}s"
            )
        else:
            print(
                f"  Fold {fold_number}/{n_folds}: "
                f"R²={metrics['R²']:.4f}, MAE={metrics['MAE']:.2f}, "
                f"treino={train_seconds:.3f}s, teste={test_seconds:.3f}s"
            )

    return summarize_results(model_name, fold_results, task)


def print_summary_table(summaries, task):
    if task == "classification":
        columns = ["Acurácia", "Precisão", "Recall", "F1-score", "Tempo treino (s)", "Tempo teste (s)"]
    else:
        columns = ["R²", "R² ajustado", "MAE", "RMSE", "Tempo treino (s)", "Tempo teste (s)"]

    print("\n" + "=" * 112)
    print("TABELA COMPARATIVA — média ± desvio padrão dos folds")
    print("=" * 112)
    header = f"{'Algoritmo':38s}" + "".join(f"{col:>22s}" for col in columns)
    print(header)
    print("-" * len(header))

    for summary in summaries:
        line = f"{summary['Algoritmo'][:37]:38s}"
        for col in columns:
            mean, std = summary[col]
            if col.endswith("(s)"):
                value = f"{mean:.4f} ± {std:.4f}"
            elif col in ("MAE", "RMSE"):
                value = f"{mean:.2f} ± {std:.2f}"
            else:
                value = f"{mean:.4f} ± {std:.4f}"
            line += f"{value:>22s}"
        print(line)


def save_summary_csv(summaries, task, output_path):
    if task == "classification":
        columns = ["Acurácia", "Precisão", "Recall", "F1-score", "Tempo treino (s)", "Tempo teste (s)"]
    else:
        columns = ["R²", "R² ajustado", "MAE", "RMSE", "Tempo treino (s)", "Tempo teste (s)"]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["Algoritmo"] + [f"{c} média" for c in columns] + [f"{c} desvio padrão" for c in columns])
        for summary in summaries:
            means = [summary[c][0] for c in columns]
            stds = [summary[c][1] for c in columns]
            writer.writerow([summary["Algoritmo"]] + means + stds)


def run_classification_experiment():
    print("\n\n### EXPERIMENTO 1 — CLASSIFICAÇÃO: BANK MARKETING ###")
    if not CLASSIFICATION_FILE.exists():
        raise FileNotFoundError(f"Não encontrei {CLASSIFICATION_FILE.name} na pasta do script.")

    X, y, target, feature_names, task, classes, total_attributes, original_rows = encode_dataset(
        CLASSIFICATION_FILE, MAX_CLASSIFICATION_ROWS
    )
    print(f"Arquivo: {CLASSIFICATION_FILE.name}")
    print(f"Instâncias no arquivo: {original_rows}")
    print(f"Amostras usadas na validação: {len(X)}")
    print(f"Atributos no ARFF (incluindo alvo): {total_attributes}")
    print(f"Variável-alvo: {target}; classes codificadas: {classes}")
    print(f"Dimensões após one-hot: {X.shape[1]}")
    print(f"Distribuição das classes na amostra: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Validação cruzada: {N_FOLDS}-fold")

    models = [
        ("kNN — Euclidiana", lambda: KNN(K_NEIGHBORS, "classification", "euclidean")),
        ("kNN — Manhattan", lambda: KNN(K_NEIGHBORS, "classification", "manhattan")),
        ("Bayesiano Gaussiano Univariado", lambda: NaiveBayes()),
        ("Bayesiano Gaussiano Multivariado", lambda: GaussianMultivariateBayes()),
    ]

    summaries = []
    for name, factory in models:
        print(f"\nModelo: {name}")
        summaries.append(evaluate_model_cv(name, factory, X, y, "classification"))

    print_summary_table(summaries, "classification")
    save_summary_csv(summaries, "classification", BASE_DIR / "resultados_classificacao.csv")
    print("\nMatrizes de confusão do último fold:")
    for summary in summaries:
        print(f"\n{summary['Algoritmo']}:\n{summary['Matriz de confusão']}")
    print("\nTabela salva em resultados_classificacao.csv")


def run_regression_experiment():
    print("\n\n### EXPERIMENTO 2 — REGRESSÃO: BRAZILIAN HOUSES TO RENT ###")
    if not REGRESSION_FILE.exists():
        raise FileNotFoundError(f"Não encontrei {REGRESSION_FILE.name} na pasta do script.")

    X, y, target, feature_names, task, classes, total_attributes, original_rows = encode_dataset(
        REGRESSION_FILE,
        MAX_REGRESSION_ROWS,
        drop_features={"hoa_(BRL)", "rent_amount_(BRL)", "property_tax_(BRL)", "fire_insurance_(BRL)"},
    )
    print(f"Arquivo: {REGRESSION_FILE.name}")
    print(f"Instâncias no arquivo: {original_rows}")
    print(f"Amostras usadas na validação: {len(X)}")
    print(f"Atributos no ARFF (incluindo alvo): {total_attributes}")
    print(f"Variável-alvo: {target}")
    print(f"Dimensões após one-hot: {X.shape[1]}")
    print("Nota: atributos que compõem diretamente o total (condomínio, aluguel, IPTU e seguro) foram excluídos para evitar vazamento do alvo.")
    print(f"Validação cruzada: {N_FOLDS}-fold")

    models = [
        ("kNN Regressão — Euclidiana", lambda: KNN(K_NEIGHBORS, "regression", "euclidean")),
        ("kNN Regressão — Manhattan", lambda: KNN(K_NEIGHBORS, "regression", "manhattan")),
        ("Regressão Linear Múltipla", lambda: MultipleLinearRegression()),
    ]

    summaries = []
    for name, factory in models:
        print(f"\nModelo: {name}")
        summaries.append(evaluate_model_cv(name, factory, X, y, "regression"))

    print_summary_table(summaries, "regression")
    save_summary_csv(summaries, "regression", BASE_DIR / "resultados_regressao.csv")
    print("\nTabela salva em resultados_regressao.csv")


def main():
    print("PROJETO DE INTELIGÊNCIA ARTIFICIAL COMPUTACIONAL")
    run_classification_experiment()
    run_regression_experiment()
    print("\nExperimentos concluídos.")


if __name__ == "__main__":
    main()