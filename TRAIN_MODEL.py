import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

df = pd.read_csv(r"C:\Users\hp\Videos\project\project 2\Bank_Customer_Churn_Project\data\European_Bank.csv")

df.drop(
    ["CustomerId","Surname"],
    axis=1,
    inplace=True
)

df["BalanceSalaryRatio"] = (
    df["Balance"] /
    (df["EstimatedSalary"] + 1)
)

df["ProductDensity"] = (
    df["NumOfProducts"] /
    (df["Tenure"] + 1)
)

df["EngagementProduct"] = (
    df["IsActiveMember"] *
    df["NumOfProducts"]
)

df["AgeTenure"] = (
    df["Age"] *
    df["Tenure"]
)

df = pd.get_dummies(
    df,
    columns=['Geography','Gender'],
    drop_first=True
)

X = df.drop("Exited",axis=1)
y = df["Exited"]
print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train,y_train)

pred = rf.predict(X_test)

prob = rf.predict_proba(X_test)[:,1]

print("Accuracy:",
      accuracy_score(y_test,pred))

print("Precision:",
      precision_score(y_test,pred))

print("Recall:",
      recall_score(y_test,pred))

print("F1:",
      f1_score(y_test,pred))

print("ROC AUC:",
      roc_auc_score(y_test,prob))


joblib.dump(
    rf,
    r"C:\Users\hp\Videos\project\project 2\Bank_Customer_Churn_Project\models\churn_model.pkl"
)

joblib.dump(
    scaler,
    r"C:\Users\hp\Videos\project\project 2\Bank_Customer_Churn_Project\models\scaler.pkl"
)


