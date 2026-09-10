import pandas as pd

file_path = "data/raw/twcs.csv"

df = pd.read_csv(file_path, nrows=5)

print("\nCOLUMNS:")
print(df.columns.tolist())

print("\nSAMPLE DATA:")
print(df.to_string())

print("\nDATA TYPES:")
print(df.dtypes)