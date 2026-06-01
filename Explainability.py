import pandas as pd

df = pd.read_csv(r"C:\Users\hp\Videos\project\project 2\Bank_Customer_Churn_Project\data\European_Bank.csv")

df.drop(
    ["CustomerId", "Surname"],
    axis=1,
    inplace=True
)

# Feature Engineering
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

# Encoding
df = pd.get_dummies(
    df,
    columns=["Geography", "Gender"],
    drop_first=True
)

X = df.drop("Exited", axis=1)
y = df["Exited"]

features = X.columns

import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load(
    r"C:\Users\hp\Videos\project\project 2\Bank_Customer_Churn_Project\models\churn_model.pkl"
)

importance = model.feature_importances_

features = X.columns

importance_df = pd.DataFrame({
    "Feature":features,
    "Importance":importance
})

importance_df = (
    importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
)

plt.figure(figsize=(10,6))

plt.barh(
    importance_df["Feature"][:10],
    importance_df["Importance"][:10]
)

plt.title(
    "Top Feature Importance"
)

plt.show()

# Create X_test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
import shap

explainer = shap.TreeExplainer(
    model
)
shap_values = (
    explainer.shap_values(X_test)
)
shap.summary_plot(
    shap_values,
    X_test
)
shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar"
)
shap.dependence_plot(
    "Age",
    shap_values,
    X_test
)

from sklearn.inspection import (
    PartialDependenceDisplay
)

PartialDependenceDisplay.from_estimator(
    model,
    X_test,
    ["Age"]
)

PartialDependenceDisplay.from_estimator(
    model,
    X_test,
    ["Balance"]
)

PartialDependenceDisplay.from_estimator(
    model,
    X_test,
    ["CreditScore"]
)

