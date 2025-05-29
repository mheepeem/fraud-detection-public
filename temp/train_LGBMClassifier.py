import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, early_stopping, plot_importance
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve
)

warnings.filterwarnings("ignore")

# === Step 1: Load Data ===
df = pd.read_parquet("datas/train_transaction_sample_cleaned_v1.parquet")
X = df.drop(columns=["isFraud"])
y = df["isFraud"]
print("✅ Data loaded:", X.shape)
print("Target distribution:\n", y.value_counts())

# === Step 2: Train/Validation Split ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Step 3: Handle Class Imbalance ===
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# === Step 4: Initialize and Train Model ===
model = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[early_stopping(100)]
)

# === Step 5: Predict Probabilities ===
y_pred_proba = model.predict_proba(X_val)[:, 1]

# === Step 6: Find Best Threshold Prioritizing Recall ≥ 0.6 ===
precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
best_threshold = 0
best_f1 = 0
target_recall = 0.60

for i, t in enumerate(thresholds):
    if recall[i] >= target_recall:
        f1 = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_index = i

print(f"\n📌 Best threshold with recall ≥ {target_recall:.2f}: {best_threshold:.4f}")
print(f"Precision: {precision[best_index]:.4f}, Recall: {recall[best_index]:.4f}, F1: {best_f1:.4f}")

# === Step 7: Final Evaluation at Best Threshold ===
y_pred = (y_pred_proba > best_threshold).astype(int)

print(f"\n🔍 Evaluation at threshold > {best_threshold:.4f}")
print("AUC:", roc_auc_score(y_val, y_pred_proba))
print("Precision:", precision_score(y_val, y_pred))
print("Recall:", recall_score(y_val, y_pred))
print("F1 Score:", f1_score(y_val, y_pred))
print("Predicted positive count:", y_pred.sum(), "/", len(y_pred))

# === Step 8: Plot Precision-Recall vs Threshold ===
plt.figure(figsize=(6, 4))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.axvline(x=best_threshold, color="red", linestyle="--", label="Best Threshold")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision-Recall vs Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Step 9: Histogram of Predicted Probabilities ===
plt.figure(figsize=(6, 4))
plt.hist(y_pred_proba, bins=100, edgecolor='k')
plt.axvline(x=best_threshold, color="red", linestyle="--", label="Best Threshold")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Probabilities")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Step 10: Feature Importance ===
plot_importance(model, max_num_features=20, importance_type="gain")
plt.title("Top 20 Important Features")
plt.tight_layout()
plt.show()
