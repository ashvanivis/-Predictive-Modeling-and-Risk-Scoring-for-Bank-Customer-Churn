
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import zipfile
# Unzip the model file
with zipfile.ZipFile('churn_model.zip', 'r') as zip_ref:
    zip_ref.extractall()
model = joblib.load(
    r"./churn_model.pkl"
)

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Bank Customer Churn Intelligence Dashboard",
    page_icon="🏦",
    layout="wide"
)


# =====================================
# CUSTOM CSS
# =====================================

# =====================================
# HEADER
# =====================================

st.markdown("""
<h1 style='text-align:center'>
🏦 Bank Customer Churn Intelligence Dashboard
</h1>

<p style='text-align:center;color:gray;font-size:18px'>
AI-Powered Customer Retention & Risk Analytics
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("👤 Customer Profile")

st.sidebar.markdown("---")

year = st.sidebar.slider("Year", 2020, 2029, 2025)

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
    [1,2,3,4]
)

has_card = st.sidebar.selectbox(
    "Has Credit Card",
    [0,1]
)

active_member = st.sidebar.selectbox(
    "Active Member",
    [0,1]
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male","Female"]
)

geography = st.sidebar.selectbox(
    "Geography",
    ["France","Germany","Spain"]
)

st.sidebar.markdown("---")

predict_btn = st.sidebar.button(
    "🚀 Predict Churn Risk",
    use_container_width=True
)

# =====================================
# TABS
# =====================================

tab1, tab2, tab3 = st.tabs(
    [
        "📊 Prediction",
        "📈 Analytics",
        "🔍 What-If Analysis"
    ]
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

if predict_btn:

    st.session_state.probability = model.predict_proba(input_df)[0][1]

    prediction = model.predict(input_df)[0]

    # =========================
    # TAB 1
    # =========================

    with tab1:

        st.subheader("Prediction Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #60A5FA, #3B82F6);
                padding:25px;
                border-radius:18px;
                box-shadow:0 8px 20px rgba(0,0,0,0.15);">
                <div style="
                    color:white;
                    font-size:15px;
                    opacity:0.9;">
                    Churn Probability
                </div>
                <div style="
                    color:white;
                    font-size:38px;
                    font-weight:bold;">
                    {st.session_state.probability:.2%}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #34D399, #10B981);
                padding:25px;
                border-radius:18px;
                box-shadow:0 8px 20px rgba(0,0,0,0.15);">
                <div style="
                    color:white;
                    font-size:15px;
                    opacity:0.9;">
                    Prediction
                </div>
                <div style="
                    color:white;
                    font-size:38px;
                    font-weight:bold;">
                    {"Churn" if prediction == 1 else "Retain"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #F472B6, #EC4899);
                padding:25px;
                border-radius:18px;
                box-shadow:0 8px 20px rgba(0,0,0,0.15);">
                <div style="
                    color:white;
                    font-size:15px;
                    opacity:0.9;">
                    Retention Score
                </div>
                <div style="
                    color:white;
                    font-size:38px;
                    font-weight:bold;">
                    {(1-st.session_state.probability):.2%}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        if st.session_state.probability < 0.30:
            st.success("🟢 Low Risk Customer")

        elif st.session_state.probability < 0.60:
            st.warning("🟠 Medium Risk Customer")

        else:
            st.error("🔴 High Risk Customer")

        st.info(f"""
    ### Customer Summary

    - Age: {age}
    - Credit Score: {credit_score}
    - Products: {products}
    - Geography: {geography}
    - Active Member: {'Yes' if active_member else 'No'}
        """)

        st.subheader("🎯 Retention Recommendations")

        if st.session_state.probability > 0.60:
            st.error("""
    • Offer Loyalty Rewards

    • Assign Relationship Manager

    • Personalized Retention Campaign
            """)

        elif st.session_state.probability > 0.30:
            st.warning("""
    • Increase Customer Engagement

    • Offer Additional Banking Products
            """)

        else:
            st.success("""
    • Customer Likely To Stay

    • Continue Standard Engagement
            """)

    # -------------------------
    # TAB 2
    # -------------------------

    with tab2:

        st.subheader("Risk Analytics")

        col1, col2 = st.columns(2)

        with col1:

            gauge_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=st.session_state.probability * 100,
                    title={"text":"Risk Score (%)"},
                    gauge={
                        "axis":{"range":[0,100]},
                        "bar":{"color":"#1E3A8A"},
                        "steps":[
                            {"range":[0,30],"color":"#D1FAE5"},
                            {"range":[30,60],"color":"#FDE68A"},
                            {"range":[60,100],"color":"#FCA5A5"}
                        ]
                    }
                )
            )

            st.plotly_chart(
                gauge_fig,
                use_container_width=True
            )

        with col2:

            prob_df = pd.DataFrame({
                "Category":["Retain","Churn"],
                "Probability":[
                    1-st.session_state.probability,
                    st.session_state.probability
                ]
            })

            pie_fig = px.pie(
                prob_df,
                names="Category",
                values="Probability",
                hole=0.55
            )

            st.plotly_chart(
                pie_fig,
                use_container_width=True
            )

        st.subheader("Top Feature Importance")

        importance_df = pd.DataFrame({
            "Feature": model.feature_names_in_,
            "Importance": model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=False
        )

        bar_fig = px.bar(
            importance_df.head(10),
            x="Importance",
            y="Feature",
            orientation="h"
        )

        bar_fig.update_layout(
            template="plotly_white",
            height=500
        )

        st.plotly_chart(
            bar_fig,
            use_container_width=True
        )

# =====================================
# WHAT IF ANALYSIS
# =====================================

with tab3:

    st.subheader("🔍 What-If Scenario Simulator")

    with st.form("what_if_form"):

        new_products = st.slider(
            "Adjust Products",
            1,
            4,
            products
        )

        new_active = st.selectbox(
            "Adjust Active Membership",
            [0,1]
        )

        run_simulation = st.form_submit_button(
            "Run Simulation"
        )

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
                scenario_probability
                - st.session_state.probability
            ) * 100

            st.metric(
                "Risk Change",
                f"{change:.2f}%"
            )
