import pandas as pd
import ipaddress

# Path to the IoT-23 connection log
file_path = "data2/opt/Malware-Project/BigDataset/IoTScenarios/CTU-IoT-Malware-Capture-8-1/bro/conn.log.labeled"

# Extract column names from the #fields header
with open(file_path, "r") as f:
    for line in f:
        if line.startswith("#fields"):
            columns = line.split()[1:]
            break

# Load connection records
df = pd.read_csv(
    file_path,
    sep=r"\s+",
    comment="#",
    names=columns,
    na_values=["-", "(empty)"],
    low_memory=False
)
# removing unnecessary columns
columns_to_drop = [
    "tunnel_parents",
    "service",
    "local_orig",
    "local_resp"
]
df = df.drop(columns=columns_to_drop)

numeric_cols = ["duration", "orig_bytes", "resp_bytes"]

for col in numeric_cols:
    df[col + "_missing"] = df[col].isna().astype(int)
    df[col] = df[col].fillna(0)

# feature engineering for IP address
def is_private_ip(ip):
    address = ipaddress.ip_address(ip)
    return int(address.is_private)

df["src_private"] = df["id.orig_h"].apply(is_private_ip)
df["dst_private"] = df["id.resp_h"].apply(is_private_ip)

print(df[[
    "id.orig_h",
    "id.resp_h",
    "src_private",
    "dst_private"
]].head(10))

# Remove columns that should not be used for training
df = df.drop(columns=[
    "uid",
    "ts",
    "history",
    "detailed-label",
    "id.orig_h",
    "id.resp_h"
])

# Convert categorical features to numerical features
df = pd.get_dummies(
    df,
    columns=["proto", "conn_state"],
    dtype=int
)

# Convert labels to binary values
df["label"] = df["label"].map({
    "Benign": 0,
    "Malicious": 1
})

# Save processed dataset
df.to_csv("data2/processed_iot23.csv", index=False)

print("Processed dataset shape:", df.shape)
print("Dataset saved successfully!")