import pandas as pd

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = "FAKE"
true["label"] = "REAL"

df = pd.concat([fake, true])

df = df[["text", "label"]]

df.to_csv("dataset.csv", index=False)

print("Dataset combined successfully!")
print("Total rows:", len(df))