import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import joblib
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve
)

warnings.filterwarnings("ignore")

# === Step 1: Load Data ===
df = pd.read_parquet("./datas/train_transaction_for_model_training.parquet")
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
model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.05,
    random_seed=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric="AUC",
    early_stopping_rounds=100,
    verbose=False
)

train_pool = Pool(X_train, y_train)
val_pool = Pool(X_val, y_val)

model.fit(train_pool, eval_set=val_pool)

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

# === Step 8: Save model and threshold ===

# Create model directory for SageMaker
os.makedirs("model", exist_ok=True)

# Save model in CatBoost native format
model.save_model("model/catboost_fraud_model.cbm")

# Save best threshold
with open("model/best_threshold.txt", "w") as f:
    f.write(str(best_threshold))

# (Optional) Save with joblib for Python-based inference
joblib.dump(model, "model/catboost_fraud_model.pkl")

# === Step 9: Package into model.tar.gz for SageMaker ===
import tarfile

with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add("model", arcname=".")

print("\n📦 Model saved and packed as model.tar.gz (ready for SageMaker deployment)")

# === Step 10: (Optional) Feature Importance Visualization ===
feature_importances = model.get_feature_importance()
top_idx = np.argsort(feature_importances)[-20:]
top_features = X.columns[top_idx]
top_importances = feature_importances[top_idx]

plt.figure(figsize=(6, 6))
plt.barh(top_features, top_importances)
plt.xlabel("Importance (Gain)")
plt.title("Top 20 Important Features")
plt.tight_layout()
plt.show()
