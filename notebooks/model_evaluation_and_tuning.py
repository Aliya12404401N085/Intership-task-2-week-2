import pandas as pd

# Load dataset
df = pd.read_csv("data/cleaned_dataset.csv")

# Display first 5 rows
print(df.head())

# Display dataset information
print(df.info())

# Display statistical summary
print(df.describe())
import pandas as pd

# Load dataset
df = pd.read_csv("data/cleaned_dataset.csv")

# Display first 5 rows
print(df.head())

# Display dataset information
print(df.info())

# Display statistical summary
print(df.describe())

# -----------------------------
# STEP 7 (paste below this)
# -----------------------------

from sklearn.model_selection import train_test_split

# Features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Convert categorical columns to numerical
X = pd.get_dummies(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("===== Logistic Regression =====")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("===== Decision Tree =====")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("===== Logistic Regression =====")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("===== Decision Tree =====")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))
