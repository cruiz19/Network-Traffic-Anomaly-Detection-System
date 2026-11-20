import pandas as pd
import ipaddress
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance

# Load the trained Random Forest and its feature names
model = joblib.load("data2/iot23_random_forest.pkl")
feature_columns = joblib.load("data2/iot23_features.pkl")

# Load a different IoT-23 capture
file_path = "data2/opt/Malware-Project/BigDataset/IoTScenarios/CTU-IoT-Malware-Capture-3-1/bro/conn.log.labeled"

# Extract column names
with open(file_path, "r") as f:
    for line in f:
        if line.startswith("#fields"):
            columns = line.split()[1:]
            break

# Read connection records
df = pd.read_csv(
    file_path,
    sep=r"\s+",
    comment="#",
    names=columns,
    na_values=["-", "(empty)"],
    low_memory=False
)

print("New dataset shape:", df.shape)

# Remove unused columns
df = df.drop(columns=[
    "tunnel_parents",
    "service",
    "local_orig",
    "local_resp"
])

# Handle missing numerical values
numeric_cols = ["duration", "orig_bytes", "resp_bytes"]

for col in numeric_cols:
    df[col + "_missing"] = df[col].isna().astype(int)
    df[col] = df[col].fillna(0)

# Engineer IP address features
def is_private_ip(ip):
    address = ipaddress.ip_address(ip)
    return int(address.is_private)

df["src_private"] = df["id.orig_h"].apply(is_private_ip)
df["dst_private"] = df["id.resp_h"].apply(is_private_ip)

# Remove identifiers and columns excluded during training
df = df.drop(columns=[
    "uid",
    "ts",
    "history",
    "detailed-label",
    "id.orig_h",
    "id.resp_h"
])

# Encode categorical features
df = pd.get_dummies(
    df,
    columns=["proto", "conn_state"],
    dtype=int
)

# Convert labels to binary
df["label"] = df["label"].map({
    "Benign": 0,
    "Malicious": 1
})

# Separate features and actual labels
X_test = df.drop(columns=["label"])
y_test = df["label"]

# Ensure the new data has exactly the features used in training
X_test = X_test.reindex(
    columns=feature_columns,
    fill_value=0
)

# Predict using the existing model (no retraining)
y_pred = model.predict(X_test)

# Evaluate performance
print("\nActual label distribution:")
print(y_test.value_counts(dropna=False))

if y_test.isna().any():
    raise ValueError("Some labels could not be mapped to Benign/Malicious.")

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Benign", "Malicious"],
    zero_division=0
))

r = permutation_importance(model,X_test,y_test,scoring="balanced_accuracy",n_repeats=5,random_state=42,n_jobs=-1)

for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(
            f"{X_test.columns[i]:<25}"
            f"{r.importances_mean[i]:.3f}"
            f" +/- {r.importances_std[i]:.3f}"
        )

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

#confusion matrix to get visual of data
cm = confusion_matrix(y_test, y_pred)
# ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BENIGN", "ATTACK"]).plot()
labels = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BENIGN", "ATTACK"])
disp.plot(include_values=False, cmap="Blues")

for i in range (len(cm)):
    for j in range (len(cm[i])):
        disp.ax_.text(
            j, i,
            f"{labels[i][j]}\n{cm[i, j]:,}",
            ha="center",
            va="center"
        )
plt.show()
