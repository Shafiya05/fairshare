import pandas as pd
from github_data import collect

df = collect("pallets/flask")
print(df.head())

summary = pd.crosstab(df["person"], df["type"])
print(summary.sort_values("commit", ascending=False).head(10))