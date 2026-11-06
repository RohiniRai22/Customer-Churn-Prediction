
import os
import joblib
import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #F4F7FC;
}

[data-testid="stSidebar"] {
    background-color: #102A43;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #102A43;
}

.subtitle {
    color: #627D98;
    font-size: 16px;
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 12px;
    border-left: 5px solid #2878D0;
    box-shadow: 0 3px 12px rgba(0,0,0,0.06);
}

.metric-label {
    color: #627D98;
    font-size: 14px;
}

.metric-value {
    color: #102A43;
    font-size: 28px;
    font-weight: bold;
}

div.stButton > button {
    background-color: #1769AA;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 25px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL AND DATA
# ==========================================

MODEL_PATH = "outputs/final_customer_churn_model.pkl"
SCALER_PATH = "outputs/final_scaler.pkl"
DATA_PATH = "outputs/encoded_customer_data.csv"
CLEAN_DATA_PATH = "outputs/cleaned_customer_data.csv"

required_files = [
    MODEL_PATH,
    SCALER_PATH,
    DATA_PATH,
    CLEAN_DATA_PATH
]

if not all(os.path.exists(path) for path in required_files):
    st.error("Required model or dataset files are missing.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

encoded_data = pd.read_csv(DATA_PATH)
customer_data = pd.read_csv(CLEAN_DATA_PATH)

feature_columns = encoded_data.drop(
    columns=["Churn"]
).columns


# ==========================================
# PREPARE CHURN DATA
# ==========================================

# Convert Yes/No into 1/0 safely
churn_numeric = (
    customer_data["Churn"]
    .astype(str)
    .str.strip()
    .map({
        "Yes": 1,
        "No": 0,
        "1": 1,
        "0": 0
    })
)

if churn_numeric.isna().any():
    st.error("Unexpected values found in the Churn column.")
    st.stop()

customer_data["Churn_Numeric"] = churn_numeric.astype(int)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("📊 Customer AI")
st.sidebar.caption(
    "Intelligent Customer Intelligence System"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Churn Prediction",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Machine Learning Model: Tuned Random Forest"
)


# ==========================================
# DASHBOARD PAGE
# ==========================================

if page == "Dashboard":

    st.markdown(
        '<p class="main-title">Customer Intelligence Dashboard</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Customer analytics and churn insights</p>',
        unsafe_allow_html=True
    )

    st.divider()

    total_customers = len(customer_data)

    churned_customers = int(
        (customer_data["Churn_Numeric"] == 1).sum()
    )

    retained_customers = int(
        (customer_data["Churn_Numeric"] == 0).sum()
    )

    churn_rate = (
        churned_customers / total_customers
    ) * 100 if total_customers > 0 else 0

    # CUSTOMER METRICS

    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        (col1, "Total Customers", f"{total_customers:,}"),
        (col2, "Churned Customers", f"{churned_customers:,}"),
        (col3, "Retained Customers", f"{retained_customers:,}"),
        (col4, "Churn Rate", f"{churn_rate:.2f}%")
    ]

    for col, label, value in metrics:

        with col:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    # CUSTOMER CHURN DISTRIBUTION

    left, right = st.columns(2)

    with left:

        st.subheader("Customer Churn Distribution")

        churn_chart = pd.DataFrame({
            "Customer Status": [
                "Stayed",
                "Churned"
            ],
            "Customers": [
                retained_customers,
                churned_customers
            ]
        })

        fig = px.pie(
            churn_chart,
            names="Customer Status",
            values="Customers",
            hole=0.55,
            title="Customer Retention Overview"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # CONTRACT ANALYSIS

    with right:

        st.subheader("Customer Contract Analysis")

        contract_chart = (
            customer_data.groupby("Contract")[
                "Churn_Numeric"
            ]
            .mean()
            .reset_index()
        )

        contract_chart["Churn Rate"] = (
            contract_chart["Churn_Numeric"] * 100
        )

        fig2 = px.bar(
            contract_chart,
            x="Contract",
            y="Churn Rate",
            text_auto=".1f",
            title="Churn Rate by Contract Type",
            labels={
                "Churn Rate": "Churn Rate (%)"
            }
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.divider()

    # DATASET PREVIEW

    st.subheader("Dataset Preview")

    st.dataframe(
        customer_data.drop(
            columns=["Churn_Numeric"]
        ).head(10),
        use_container_width=True
    )


# ==========================================
# CHURN PREDICTION PAGE
# ==========================================

elif page == "Churn Prediction":

    st.markdown(
        '<p class="main-title">Customer Churn Prediction</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Enter customer details to predict churn probability</p>',
        unsafe_allow_html=True
    )

    st.divider()

    with st.form("customer_prediction_form"):

        # PERSONAL INFORMATION

        st.subheader("1. Personal Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

        with col2:
            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x:
                "Yes" if x == 1 else "No"
            )

        with col3:
            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        st.divider()

        # SERVICE INFORMATION

        st.subheader("2. Service Information")

        col1, col2, col3 = st.columns(3)

        with col1:

            phone_service = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )

        with col2:

            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"]
            )

        with col3:

            online_backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"]
            )

            device_protection = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"]
            )

        col1, col2, col3 = st.columns(3)

        with col1:

            tech_support = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"]
            )

        with col2:

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"]
            )

        with col3:

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"]
            )

        st.divider()

        # CONTRACT AND BILLING

        st.subheader("3. Contract and Billing")

        col1, col2 = st.columns(2)

        with col1:

            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )

            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

        with col2:

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        st.divider()

        # CHARGES AND TENURE

        st.subheader("4. Charges and Tenure")

        col1, col2, col3 = st.columns(3)

        with col1:

            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=72,
                value=12
            )

        with col2:

            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=200.0,
                value=85.50
            )

        with col3:

            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=1026.0
            )

        submitted = st.form_submit_button(
            "🔍 Predict Customer Churn",
            use_container_width=True
        )

    # ======================================
    # PREDICTION
    # ======================================

    if submitted:

        customer = {

            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges

        }

        customer_df = pd.DataFrame([customer])

        # ENCODING

        customer_encoded = pd.get_dummies(
            customer_df,
            drop_first=True,
            dtype=int
        )

        customer_encoded = customer_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # SCALING

        numeric_columns = [
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]

        customer_encoded[numeric_columns] = (
            scaler.transform(
                customer_encoded[numeric_columns]
            )
        )

        # MODEL PREDICTION

        prediction = model.predict(
            customer_encoded
        )[0]

        probability = model.predict_proba(
            customer_encoded
        )[0][1]

        st.divider()

        st.subheader("5. Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:

                st.error(
                    "⚠️ Customer is likely to CHURN"
                )

            else:

                st.success(
                    "✅ Customer is likely to STAY"
                )

        with col2:

            st.metric(
                "Estimated Churn Probability",
                f"{probability * 100:.2f}%"
            )

        st.progress(float(probability))

        st.caption(
            "This probability is the model's estimated likelihood "
            "of churn, not a guarantee of future customer behavior."
        )


# ==========================================
# ABOUT PROJECT PAGE
# ==========================================

elif page == "About Project":

    st.markdown(
        '<p class="main-title">About the Project</p>',
        unsafe_allow_html=True
    )

    st.write("""
    **Intelligent Customer Intelligence & Predictive Analytics System**

    This project uses machine learning to predict whether a
    telecommunications customer is likely to leave the service.

    **Dataset:** IBM Telco Customer Churn

    **Selected Algorithm:** Tuned Random Forest Classifier
    """)

    st.subheader("Model Evaluation")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", "75.87%")
    col2.metric("Precision", "53.14%")
    col3.metric("Recall", "77.01%")

    st.metric("F1 Score", "62.88%")

    st.subheader("Technologies Used")

    st.write("""
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Random Forest
    - Streamlit
    - Plotly
    """)

    st.info(
        "The Tuned Random Forest model was selected "
        "based on its highest F1 score among the evaluated models."
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Intelligent Customer Intelligence & Predictive Analytics System | Machine Learning Project"
)
