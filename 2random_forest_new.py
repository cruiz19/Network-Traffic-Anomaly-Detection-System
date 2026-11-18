import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load processed dataset
df = pd.read_csv("data2/processed_iot23.csv")

# Separate features and labels
X = df.drop(columns=["label"])
y = df["label"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

#trying on new file
joblib.dump(model, "data2/iot23_random_forest.pkl")
joblib.dump(X_train.columns.tolist(), "data2/iot23_features.pkl")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Benign", "Malicious"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#confusion matrix to get visual of data
cm = confusion_matrix(y_test, y_pred)
# ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BENIGN", "ATTACK"]).plot()
labels = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BENIGN", "ATTACK"])
disp.plot(include_values=False)

for i in range (len(cm)):
    for j in range (len(cm[i])):
        disp.ax_.text(
            j, i,
            f"{labels[i][j]}\n{cm[i, j]:,}",
            ha="center",
            va="center"
        )
plt.show()