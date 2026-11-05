import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

#sanity check of data shapes
train_feat = X_train.shape
test_feat =X_test.shape
train_label = y_train.shape
test_label = y_test.shape

if train_feat[0] != train_label[0]:
    print("Training features and labels rows do not match!!")
elif test_feat[0] != test_label[0]:
    print("Testing features and labels do not match!!")

# creating random forest classifier model and training it with the training data
rf = RandomForestClassifier()
rf.fit(X_train, y_train) #adjusts parameters of the model to best fit the training data

#now need to test the model with the testing data
y_pred = rf.predict(X_test)
#test accuracy
accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

#print report to get more metrics
print(classification_report(y_test, y_pred))

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
