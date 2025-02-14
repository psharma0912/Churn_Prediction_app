import streamlit as st
import pandas as pd
import joblib
import warnings;
warnings.filterwarnings('ignore')

# Load the trained model and scaler
model = joblib.load('churn_prediction.joblib')
scaler = joblib.load('scaler.joblib')

# List of all features in the model
all_features = [
    'MonthlyCharges', 'TotalCharges', 'tenure', 'gender_Male', 'SeniorCitizen_1',
    'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

# Streamlit app
st.title("Customer Churn Prediction")

# Input form for user data
st.header("Enter Customer Information")
with st.form("churn_form"):
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, step=0.01, value=29.85)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, step=0.01, value=49.605)
    tenure = st.number_input("Tenure (Months)", min_value=0, step=1, value=60)
    gender_Male = st.selectbox("Gender (0 = Female, 1 = Male)", [0, 1], index=0)
    SeniorCitizen_1 = st.selectbox("Senior Citizen (0 = No, 1 = Yes)", [0, 1], index=0)
    Partner_Yes = st.selectbox("Has Partner (0 = No, 1 = Yes)", [0, 1], index=1)
    Dependents_Yes = st.selectbox("Has Dependents (0 = No, 1 = Yes)", [0, 1], index=0)
    PhoneService_Yes = st.selectbox("Has Phone Service (0 = No, 1 = Yes)", [0, 1], index=0)
    MultipleLines_No_phone_service = st.selectbox("No Phone Service (0 = No, 1 = Yes)", [0, 1], index=1)
    MultipleLines_Yes = st.selectbox("Multiple Lines (0 = No, 1 = Yes)", [0, 1], index=0)
    InternetService_Fiber_optic = st.selectbox("Fiber Optic Internet (0 = No, 1 = Yes)", [0, 1], index=0)
    InternetService_No = st.selectbox("No Internet Service (0 = No, 1 = Yes)", [0, 1], index=0)
    OnlineSecurity_No_internet_service = st.selectbox("No Internet Security (0 = No, 1 = Yes)", [0, 1], index=0)
    OnlineSecurity_Yes = st.selectbox("Has Online Security (0 = No, 1 = Yes)", [0, 1], index=0)
    OnlineBackup_No_internet_service = st.selectbox("No Online Backup (0 = No, 1 = Yes)", [0, 1], index=0)
    OnlineBackup_Yes = st.selectbox("Has Online Backup (0 = No, 1 = Yes)", [0, 1], index=1)
    DeviceProtection_No_internet_service = st.selectbox("No Device Protection (0 = No, 1 = Yes)", [0, 1], index=0)
    DeviceProtection_Yes = st.selectbox("Has Device Protection (0 = No, 1 = Yes)", [0, 1], index=0)
    TechSupport_No_internet_service = st.selectbox("No Tech Support (0 = No, 1 = Yes)", [0, 1], index=0)
    TechSupport_Yes = st.selectbox("Has Tech Support (0 = No, 1 = Yes)", [0, 1], index=0)
    StreamingTV_No_internet_service = st.selectbox("No Streaming TV (0 = No, 1 = Yes)", [0, 1], index=0)
    StreamingTV_Yes = st.selectbox("Has Streaming TV (0 = No, 1 = Yes)", [0, 1], index=0)
    StreamingMovies_No_internet_service = st.selectbox("No Streaming Movies (0 = No, 1 = Yes)", [0, 1], index=0)
    StreamingMovies_Yes = st.selectbox("Has Streaming Movies (0 = No, 1 = Yes)", [0, 1], index=0)
    Contract_One_year = st.selectbox("One-Year Contract (0 = No, 1 = Yes)", [0, 1], index=0)
    Contract_Two_year = st.selectbox("Two-Year Contract (0 = No, 1 = Yes)", [0, 1], index=0)
    PaperlessBilling_Yes = st.selectbox("Paperless Billing (0 = No, 1 = Yes)", [0, 1], index=1)
    PaymentMethod_Credit_card = st.selectbox("Credit Card Payment (0 = No, 1 = Yes)", [0, 1], index=0)
    PaymentMethod_Electronic_check = st.selectbox("Electronic Check Payment (0 = No, 1 = Yes)", [0, 1], index=1)
    PaymentMethod_Mailed_check = st.selectbox("Mailed Check Payment (0 = No, 1 = Yes)", [0, 1], index=0)

    # Submit button
    submitted = st.form_submit_button("Predict")

# If the form is submitted
if submitted:
    # Create a dictionary with the input data
    data = {
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'tenure': tenure,
        'gender_Male': gender_Male,
        'SeniorCitizen_1': SeniorCitizen_1,
        'Partner_Yes': Partner_Yes,
        'Dependents_Yes': Dependents_Yes,
        'PhoneService_Yes': PhoneService_Yes,
        'MultipleLines_No phone service': MultipleLines_No_phone_service,
        'MultipleLines_Yes': MultipleLines_Yes,
        'InternetService_Fiber optic': InternetService_Fiber_optic,
        'InternetService_No': InternetService_No,
        'OnlineSecurity_No internet service': OnlineSecurity_No_internet_service,
        'OnlineSecurity_Yes': OnlineSecurity_Yes,
        'OnlineBackup_No internet service': OnlineBackup_No_internet_service,
        'OnlineBackup_Yes': OnlineBackup_Yes,
        'DeviceProtection_No internet service': DeviceProtection_No_internet_service,
        'DeviceProtection_Yes': DeviceProtection_Yes,
        'TechSupport_No internet service': TechSupport_No_internet_service,
        'TechSupport_Yes': TechSupport_Yes,
        'StreamingTV_No internet service': StreamingTV_No_internet_service,
        'StreamingTV_Yes': StreamingTV_Yes,
        'StreamingMovies_No internet service': StreamingMovies_No_internet_service,
        'StreamingMovies_Yes': StreamingMovies_Yes,
        'Contract_One year': Contract_One_year,
        'Contract_Two year': Contract_Two_year,
        'PaperlessBilling_Yes': PaperlessBilling_Yes,
        'PaymentMethod_Credit card (automatic)': PaymentMethod_Credit_card,
        'PaymentMethod_Electronic check': PaymentMethod_Electronic_check,
        'PaymentMethod_Mailed check': PaymentMethod_Mailed_check
    }

    # Convert to DataFrame
    input_df = pd.DataFrame(data, index=[0])
    input_df = input_df[all_features]  # Ensure column order matches training data

    # Scale the input data
    scaled_input = scaler.transform(input_df)

    # Make prediction and display the result
    probabilities = model.predict_proba(scaled_input)
    churn_probability = probabilities[0][1]

    st.subheader(f"Probability of churn: {churn_probability :.2f}%")

    if churn_probability > 0.5:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is not likely to churn.")
