import pandas as pd
from sklearn.model_selection import train_test_split

#load one of the csv files from CICIDS2017 dataset
df = pd.read_csv("data/Wednesday-workingHours.pcap_ISCX.csv")

# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

#benign = 0, malicious (everything else) = 1
df["Malicious"] = (df["Label"] != "BENIGN").astype(int)
print(df["Malicious"].value_counts())

#features used by first model
features = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
]

#x contains the information to make a prediction
X = df[features]

#y contains the correct answer
Y = df["Malicious"]

# print(X.head())
# print(Y.head())

#split x and y and create four variables:
    # X_train = features the model LEARNS from
    # Y_train = correct answers for those training examples
    # X_test  = features we TEST the model with
    # Y_test  = correct answers used to check its predictions
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining labels:")
print(Y_train.value_counts())

print("\nTesting labels:")
print(Y_test.value_counts())