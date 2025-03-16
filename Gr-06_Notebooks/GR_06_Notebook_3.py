import streamlit as st
import requests

# Title and description
st.title("Credit Risk Prediction")
st.write("Enter customer details to predict their credit rating (Good or Bad).")

# Numeric Inputs
duration = st.number_input("Duration (months)", min_value=1, value=12)
credit_amount = st.number_input("Credit Amount", min_value=1, value=1000)
installment_rate = st.number_input("Installment Rate (%)", min_value=1, max_value=4, value=2)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
present_residence = st.number_input("Present Residence (years)", min_value=1, max_value=4, value=2)
number_of_existing_credits = st.number_input("Number of Existing Credits", min_value=1, max_value=4, value=1)
dependents = st.number_input("Number of Dependents", min_value=1, max_value=2, value=1)

# Categorical Inputs
checking_account_status = st.selectbox("Checking Account Status", ["A11", "A12", "A13", "A14"])
credit_history = st.selectbox("Credit History", ["A30", "A31", "A32", "A33", "A34"])
purpose = st.selectbox("Purpose", ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410"])
savings_account = st.selectbox("Savings Account", ["A61", "A62", "A63", "A64", "A65"])
employment_status = st.selectbox("Employment Status", ["A71", "A72", "A73", "A74", "A75"])
personal_status_sex = st.selectbox("Personal Status & Sex", ["A91", "A92", "A93", "A94"])
other_debtors = st.selectbox("Other Debtors", ["A101", "A102", "A103"])
property = st.selectbox("Property", ["A121", "A122", "A123", "A124"])
other_installment_plans = st.selectbox("Other Installment Plans", ["A141", "A142", "A143"])
housing = st.selectbox("Housing", ["A151", "A152", "A153"])
job = st.selectbox("Job", ["A171", "A172", "A173", "A174"])
telephone = st.selectbox("Telephone", ["A191", "A192"])
foreign_worker = st.selectbox("Foreign Worker", ["A201", "A202"])

# Prepare data for API call
data = {
    "checking_account_status": checking_account_status,
    "duration": duration,
    "credit_history": credit_history,
    "purpose": purpose,
    "credit_amount": credit_amount,
    "savings_account": savings_account,
    "employment_status": employment_status,
    "installment_rate": installment_rate,
    "personal_status_sex": personal_status_sex,
    "other_debtors": other_debtors,
    "present_residence": present_residence,
    "property": property,
    "age": age,
    "other_installment_plans": other_installment_plans,
    "housing": housing,
    "number_of_existing_credits": number_of_existing_credits,
    "job": job,
    "dependents": dependents,
    "telephone": telephone,
    "foreign_worker": foreign_worker,
}

# Button to trigger prediction
if st.button("Predict"):
    # Endpoint URL (ensure FastAPI is running at this URL)
    url = "http://127.0.0.1:8000/predict"

    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            prediction = result.get("prediction", "No prediction returned")
            st.success(f"Prediction: {prediction}")
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
