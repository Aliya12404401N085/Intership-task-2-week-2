import pandas as pd

# Load dataset
df = pd.read_csv("data/cleaned_dataset.csv")

# Display first 5 rows
print(df.head())

# Display dataset information
print(df.info())

# Display statistical summary
print(df.describe())
