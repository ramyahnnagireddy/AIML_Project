# W4D5: 1M Capstone - Sentiment Classifier
# Logistic Regression vs Random Forest
# Cynaris Internship - AIML

import os
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)


# --------------------------------------------------
# 1. Create output directory
# --------------------------------------------------

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. Load Binary Classification Dataset
# --------------------------------------------------

data = load_breast_cancer()

X = data.data
y = data.target

print("=" * 60)
print("W4D5 - BINARY CLASSIFICATION")
print("=" * 60)

print(f"Dataset: Breast Cancer Wisconsin")
print(f"Samples: {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {data.target_names}")
print()


# --------------------------------------------------
# 3. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print()


# --------------------------------------------------
# 4. Logistic Regression
# --------------------------------------------------

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=5000,
        random_state=42
    ))
])

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)
logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1]


print("=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        logistic_predictions,
        target_names=data.target_names
    )
)


logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

logistic_precision = precision_score(
    y_test,
    logistic_predictions
)

logistic_recall = recall_score(
    y_test,
    logistic_predictions
)

logistic_roc_auc = roc_auc_score(
    y_test,
    logistic_probabilities
)


print(f"Accuracy : {logistic_accuracy:.4f}")
print(f"Precision: {logistic_precision:.4f}")
print(f"Recall   : {logistic_recall:.4f}")
print(f"ROC-AUC  : {logistic_roc_auc:.4f}")
print()


# --------------------------------------------------
# 5. Logistic Regression Confusion Matrix
# --------------------------------------------------

logistic_cm = confusion_matrix(
    y_test,
    logistic_predictions
)

logistic_display = ConfusionMatrixDisplay(
    confusion_matrix=logistic_cm,
    display_labels=data.target_names
)

logistic_display.plot()

plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "logistic_confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# --------------------------------------------------
# 6. Logistic Regression ROC-AUC Curve
# --------------------------------------------------

fpr_logistic, tpr_logistic, _ = roc_curve(
    y_test,
    logistic_probabilities
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr_logistic,
    tpr_logistic,
    label=(
        f"Logistic Regression "
        f"(AUC = {logistic_roc_auc:.4f})"
    )
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression - ROC-AUC Curve")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "logistic_roc_auc.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# --------------------------------------------------
# 7. Random Forest Classifier
# --------------------------------------------------

random_forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

random_forest_model.fit(
    X_train,
    y_train
)

rf_predictions = random_forest_model.predict(
    X_test
)

rf_probabilities = random_forest_model.predict_proba(
    X_test
)[:, 1]


print("=" * 60)
print("RANDOM FOREST CLASSIFIER")
print("=" * 60)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        rf_predictions,
        target_names=data.target_names
    )
)


rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

rf_precision = precision_score(
    y_test,
    rf_predictions
)

rf_recall = recall_score(
    y_test,
    rf_predictions
)

rf_roc_auc = roc_auc_score(
    y_test,
    rf_probabilities
)


print(f"Accuracy : {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall   : {rf_recall:.4f}")
print(f"ROC-AUC  : {rf_roc_auc:.4f}")
print()


# --------------------------------------------------
# 8. Random Forest Confusion Matrix
# --------------------------------------------------

rf_cm = confusion_matrix(
    y_test,
    rf_predictions
)

rf_display = ConfusionMatrixDisplay(
    confusion_matrix=rf_cm,
    display_labels=data.target_names
)

rf_display.plot()

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "random_forest_confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# --------------------------------------------------
# 9. Model Comparison
# --------------------------------------------------

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    f"{'Metric':<15}"
    f"{'Logistic Regression':<22}"
    f"{'Random Forest':<15}"
)

print("-" * 52)

print(
    f"{'Accuracy':<15}"
    f"{logistic_accuracy:<22.4f}"
    f"{rf_accuracy:<15.4f}"
)

print(
    f"{'Precision':<15}"
    f"{logistic_precision:<22.4f}"
    f"{rf_precision:<15.4f}"
)

print(
    f"{'Recall':<15}"
    f"{logistic_recall:<22.4f}"
    f"{rf_recall:<15.4f}"
)

print(
    f"{'ROC-AUC':<15}"
    f"{logistic_roc_auc:<22.4f}"
    f"{rf_roc_auc:<15.4f}"
)

print()


# --------------------------------------------------
# 10. Combined ROC-AUC Comparison
# --------------------------------------------------

fpr_rf, tpr_rf, _ = roc_curve(
    y_test,
    rf_probabilities
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr_logistic,
    tpr_logistic,
    label=(
        f"Logistic Regression "
        f"(AUC = {logistic_roc_auc:.4f})"
    )
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label=(
        f"Random Forest "
        f"(AUC = {rf_roc_auc:.4f})"
    )
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Comparison")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "roc_auc_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# --------------------------------------------------
# 11. Final Result
# --------------------------------------------------

print("=" * 60)
print("W4D5 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nSaved graphs:")

print("1. outputs/logistic_confusion_matrix.png")
print("2. outputs/logistic_roc_auc.png")
print("3. outputs/random_forest_confusion_matrix.png")
print("4. outputs/roc_auc_comparison.png")

print("\nBest overall model: Logistic Regression")
print(f"Logistic Regression Accuracy: {logistic_accuracy:.4f}")
print(f"Random Forest Accuracy:       {rf_accuracy:.4f}")

print("\nAll graphs saved successfully.")
