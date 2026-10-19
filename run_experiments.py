import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from grader_contracts.linear_models import RegressionInput, ClassificationInput
from linear_models import (
    train_linear_regression, train_logistic_classifier,
    compare_classical_models, get_metrics, scale_params,
)

SEED = 42
results = {}

df = pd.read_csv("Student_Performance.csv")
df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})
X = df.drop(columns=["Performance Index"]).values
y = df["Performance Index"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
reg = train_linear_regression(RegressionInput(X_tr, y_tr, X_te, y_te))
mse = mean_squared_error(y_te, reg.predictions)
print("Линейная регрессия: MSE test =", mse)
print("MSE на эпохах 1, 10, 100, 400, 800:", [round(reg.loss_history[i - 1], 3) for i in (1, 10, 100, 400, 800)])
print("веса:", reg.weights, "bias:", reg.bias)
results["linear_regression"] = {"test_mse": float(mse)}
print("разные learning rate (800 эпох)")

for lr in (0.001, 0.01, 0.05, 0.1, 0.5):
    r = train_linear_regression(RegressionInput(X_tr, y_tr, X_te, y_te, learning_rate=lr))
    print(lr, mean_squared_error(y_te, r.predictions))

print("разное число эпох (lr=0.05)")

for ep in (10, 50, 100, 800):
    r = train_linear_regression(RegressionInput(X_tr, y_tr, X_te, y_te, epochs=ep))
    print(ep, mean_squared_error(y_te, r.predictions))

g = pd.read_csv("german.csv", sep=";")
Xg = g.drop(columns=["Creditability"]).values
yg = g["Creditability"].values
Xg_tr, Xg_te, yg_tr, yg_te = train_test_split(Xg, yg, test_size=0.2, random_state=SEED, stratify=yg)
cdata = ClassificationInput(Xg_tr, yg_tr, Xg_te, yg_te)
clf = train_logistic_classifier(cdata)
own = get_metrics(yg_te, clf.predictions, clf.probabilities)
print("\nСобственная логистическая регрессия:", own)
print("BCE на эпохах 1, 10, 100, 500, 1000:", [round(clf.loss_history[i - 1], 4) for i in (1, 10, 100, 500, 1000)])
print("доля класса 1 в test:", yg_te.mean(), " доля предсказанных 1:", clf.predictions.mean())
print("разные learning rate (1000 эпох)")

for lr in (0.001, 0.01, 0.1, 0.5):
    r = train_logistic_classifier(ClassificationInput(Xg_tr, yg_tr, Xg_te, yg_te, learning_rate=lr))
    m = get_metrics(yg_te, r.predictions, r.probabilities)
    print(lr, "loss=%.4f auc=%.4f acc=%.3f f1=%.3f" % (r.loss_history[-1], m.roc_auc, m.accuracy, m.f1))

print("разное число эпох (lr=0.1)")

for ep in (10, 100, 1000):
    r = train_logistic_classifier(ClassificationInput(Xg_tr, yg_tr, Xg_te, yg_te, epochs=ep))
    m = get_metrics(yg_te, r.predictions, r.probabilities)
    print(ep, "loss=%.4f auc=%.4f acc=%.3f f1=%.3f" % (r.loss_history[-1], m.roc_auc, m.accuracy, m.f1))

cmp = compare_classical_models(cdata)
print("\nСравнение:")
print("own ", cmp.logistic_regression)
print("tree ", cmp.decision_tree)
print("knn ", cmp.knn)

mean, std = scale_params(Xg_tr.astype(float))
sk = LogisticRegression(max_iter=1000).fit((Xg_tr - mean) / std, yg_tr)
Xs_te = (Xg_te - mean) / std
sk_m = get_metrics(yg_te, sk.predict(Xs_te), sk.predict_proba(Xs_te)[:, 1])

print("дерево: разная глубина")

for d in (None, 3, 5, 8):
    t = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(Xg_tr, yg_tr)
    m = get_metrics(yg_te, t.predict(Xg_te), t.predict_proba(Xg_te)[:, 1])
    print(d, "auc=%.4f acc=%.3f" % (m.roc_auc, m.accuracy))

print("KNN: разное k")

for k in (3, 5, 11, 21):
    kn = KNeighborsClassifier(n_neighbors=k).fit((Xg_tr - mean) / std, yg_tr)
    m = get_metrics(yg_te, kn.predict(Xs_te), kn.predict_proba(Xs_te)[:, 1])
    print(k, "auc=%.4f acc=%.3f" % (m.roc_auc, m.accuracy))

results["classification"] = {
    "own_logistic": vars(own),
    "logistic_regression": vars(sk_m),
    "decision_tree": vars(cmp.decision_tree),
    "knn": vars(cmp.knn),
}
with open("experiment_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
