import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

SAVE_PATH = r"C:\Users\KATHERINE\OneDrive\Desktop\chrun_app"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

DATA_PATH = r"C:\Users\KATHERINE\OneDrive\Desktop\churn_pred\churn.csv"

df = pd.read_csv(DATA_PATH)

if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

df.dropna(inplace=True)

target_col = 'Churn'

le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])

df = pd.get_dummies(df, drop_first=True)

X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

feature_names = X.columns.tolist()

pickle.dump(feature_names, open(os.path.join(SAVE_PATH, "feature_names.pkl"), "wb"))
pickle.dump(model, open(os.path.join(SAVE_PATH, "churn_model.pkl"), "wb"))
pickle.dump(scaler, open(os.path.join(SAVE_PATH, "scaler.pkl"), "wb"))

print("\nModel, scaler, and feature names saved successfully!")
