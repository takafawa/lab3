import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)

def scale_params(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return mean, std

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    x_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).ravel()
    x_test = np.asarray(data.test_features, dtype=float)
    mean, std = scale_params(x_train)
    x_train_s = (x_train - mean) / std
    n, k = x_train_s.shape
    w = np.zeros(k)
    b = 0.0
    loss_history = []

    for _ in range(data.epochs):
        pred = x_train_s @ w + b
        error = pred - y_train
        loss_history.append(float(np.mean(error ** 2)))
        grad_w = 2 / n * (x_train_s.T @ error)
        grad_b = 2 / n * error.sum()
        w = w - data.learning_rate * grad_w
        b = b - data.learning_rate * grad_b

    weights = w / std
    bias = b - np.sum(w * mean / std)
    predictions = x_test @ weights + bias

    return RegressionResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        predictions=predictions,
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    x_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).ravel()
    x_test = np.asarray(data.test_features, dtype=float)

    mean, std = scale_params(x_train)
    x_train_s = (x_train - mean) / std

    n, k = x_train_s.shape
    w = np.zeros(k)
    b = 0.0
    loss_history = []
    eps = 1e-12

    for _ in range(data.epochs):
        p = sigmoid(x_train_s @ w + b)
        loss = -np.mean(y_train * np.log(p + eps) + (1 - y_train) * np.log(1 - p + eps))
        loss_history.append(float(loss))
        error = p - y_train
        grad_w = x_train_s.T @ error / n
        grad_b = error.mean()
        w = w - data.learning_rate * grad_w
        b = b - data.learning_rate * grad_b

    weights = w / std
    bias = b - np.sum(w * mean / std)
    probabilities = sigmoid(x_test @ weights + bias)
    predictions = (probabilities >= 0.5).astype(int)

    return ClassificationResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        probabilities=probabilities,
        predictions=predictions,
    )


def get_metrics(y_true, y_pred, y_proba):
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred)),
        recall=float(recall_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred)),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    x_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target).ravel()
    x_test = np.asarray(data.test_features, dtype=float)
    y_test = np.asarray(data.test_target).ravel()
    own = train_logistic_classifier(data)
    own_metrics = get_metrics(y_test, own.predictions, own.probabilities)
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(x_train, y_train)
    tree_metrics = get_metrics(y_test, tree.predict(x_test), tree.predict_proba(x_test)[:, 1])
    mean, std = scale_params(x_train)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit((x_train - mean) / std, y_train)
    x_test_s = (x_test - mean) / std
    knn_metrics = get_metrics(y_test, knn.predict(x_test_s), knn.predict_proba(x_test_s)[:, 1])

    return BaselineComparisonResult(
        logistic_regression=own_metrics,
        decision_tree=tree_metrics,
        knn=knn_metrics,
    )
