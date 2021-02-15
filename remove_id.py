import pandas as pd

df = pd.read_csv("submission.csv")
print(df.head())
print(df.info())
del df["Unnamed: 0"]
print(df.head())
print(df.info())
df.to_csv("submission.csv", index=False)