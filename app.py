
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
model = joblib.load(
    r"C:\Users\hp\Videos\project\project 2\Bank_Customer_Churn_Project\models\churn_model.pkl"
)

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# Initialize session state for probability
if "probability" not in st.session_state:
    st.session_state.probability = None

# =====================================
# HEADER
# =====================================

st.title("🏦 Bank Customer Churn Prediction Dashboard")
st.markdown(
    "Predict customer churn risk and analyze factors influencing customer retention."
)

# =====================================
# SIDEBAR INPUTS
# =====================================

st.sidebar.header("Customer Information")

year = st.sidebar.slider(
    "Year",
    2020,
    2023,
    2021
)

credit_score = st.sidebar.slider(
    "Credit Score",
    300,
    900,
    650
)

age = st.sidebar.slider(
    "Age",
    18,
    90,
    35
)

tenure = st.sidebar.slider(
    "Tenure",
    0,
    10,
    5
)

balance = st.sidebar.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

salary = st.sidebar.number_input(
    "Estimated Salary",
    min_value=1000.0,
    value=50000.0
)

products = st.sidebar.selectbox(
    "Number of Products",
    [1, 2, 3, 4]
)

has_card = st.sidebar.selectbox(
    "Has Credit Card",
    [0, 1]
)

active_member = st.sidebar.selectbox(
    "Active Member",
    [0, 1]
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

geography = st.sidebar.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

# =====================================
# FEATURE ENGINEERING
# =====================================

balance_salary_ratio = balance / (salary + 1)

product_density = products / (tenure + 1)

engagement_product = active_member * products

age_tenure = age * tenure

# =====================================
# ENCODING
# =====================================

geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

gender_male = 1 if gender == "Male" else 0

# =====================================
# INPUT DATAFRAME
# =====================================

input_df = pd.DataFrame({
    "Year":[year],
    "CreditScore":[credit_score],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balance],
    "NumOfProducts":[products],
    "HasCrCard":[has_card],
    "IsActiveMember":[active_member],
    "EstimatedSalary":[salary],

    "BalanceSalaryRatio":[balance_salary_ratio],
    "ProductDensity":[product_density],
    "EngagementProduct":[engagement_product],
    "AgeTenure":[age_tenure],

    "Geography_Germany":[geo_germany],
    "Geography_Spain":[geo_spain],
    "Gender_Male":[gender_male]

})

# =====================================
# PREDICTION
# =====================================

if st.button("Predict Churn Risk"):

    st.session_state.probability = model.predict_proba(input_df)[0][1]

    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Churn Probability",
            f"{st.session_state.probability:.2%}"
        )

    with col2:

        st.metric(
            "Binary Prediction",
            "Churn" if prediction == 1 else "Retain"
        )

    # =====================================
    # RISK LEVEL
    # =====================================

    if st.session_state.probability < 0.30:
        st.success("🟢 Low Risk Customer")

    elif st.session_state.probability < 0.60:
        st.warning("🟠 Medium Risk Customer")

    else:
        st.error("🔴 High Risk Customer")

    # =====================================
    # GAUGE CHART
    # =====================================

    st.subheader("Churn Risk Score")

    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=st.session_state.probability * 100,

            title={"text": "Risk Score (%)"},

            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 30], "color": "lightgreen"},
                    {"range": [30, 60], "color": "orange"},
                    {"range": [60, 100], "color": "red"}
                ]
            }
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================
    # PROBABILITY DISTRIBUTION
    # =====================================

    st.subheader("Probability Distribution")

    prob_df = pd.DataFrame({

        "Category":["Retain", "Churn"],

        "Probability":[
            1 - st.session_state.probability,
            st.session_state.probability
        ]
    })

    fig2 = px.pie(
        prob_df,
        names="Category",
        values="Probability",
        hole=0.5,
        title="Retention vs Churn Probability"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # =====================================
    # FEATURE IMPORTANCE
    # =====================================

    st.subheader("Feature Importance Dashboard")

    importance_df = pd.DataFrame({

        "Feature": model.feature_names_in_,

        "Importance": model.feature_importances_

    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    fig3 = px.bar(
        importance_df.head(10),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 10 Important Features"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # =====================================
# WHAT-IF ANALYSIS
# =====================================

st.subheader("What-If Scenario Simulator")

# Initialize session state only once
if "new_products" not in st.session_state:
    st.session_state.new_products = int(products)

if "new_active" not in st.session_state:
    st.session_state.new_active = int(active_member)

with st.form("what_if_form"):

    new_products = st.slider(
        "Adjust Number of Products",
        min_value=1,
        max_value=4,
        value=st.session_state.new_products
    )

    new_active = st.selectbox(
        "Adjust Active Membership",
        options=[0, 1],
        index=st.session_state.new_active
    )

    run_simulation = st.form_submit_button(
        "Run Simulation"
    )

# Save selected values
st.session_state.new_products = new_products
st.session_state.new_active = new_active

# Run prediction
if run_simulation:

    scenario_df = input_df.copy()

    scenario_df["NumOfProducts"] = new_products
    scenario_df["IsActiveMember"] = new_active

    scenario_df["ProductDensity"] = (
        new_products / (tenure + 1)
    )

    scenario_df["EngagementProduct"] = (
        new_products * new_active
    )

    scenario_probability = model.predict_proba(
        scenario_df
    )[0][1]

    st.metric(
        "Updated Churn Probability",
        f"{scenario_probability:.2%}"
    )
    
    if st.session_state.probability is not None:
        change = (
            scenario_probability - st.session_state.probability
        ) * 100

        st.write(
            f"Change in Risk: {change:.2f}%"
        )
    else:
        st.warning("Please run the initial prediction first to see the change in risk.")

# Debug (remove later)
st.write("Current Products:", st.session_state.new_products)
st.write("Current Active Member:", st.session_state.new_active)