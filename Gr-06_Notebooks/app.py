import streamlit as st
import requests
import pickle
import pandas as pd



# Title and description
st.title("Credit Rating Prediction")
st.write("Enter client details to predict their credit rating (Good/Bad).")

# Categorical Inputs with Descriptions
checking_account_status = st.selectbox(
    "Status of Existing Checking Account",
    ["A11: < 0 DM", "A12: 0 <= … < 200 DM", "A13: >= 200 DM / salary assignments", "A14: No checking account"]
)
credit_history = st.selectbox(
    "Credit History",
    ["A30: No credits taken / all paid back duly", "A31: All credits at this bank paid back duly",
     "A32: Existing credits paid back duly till now", "A33: Delay in paying off in the past",
     "A34: Critical account / other credits exist"]
)
purpose = st.selectbox(
    "Purpose",
    ["A40: Car (new)", "A41: Car (used)", "A42: Furniture/equipment", "A43: Radio/television",
     "A44: Domestic appliances", "A45: Repairs", "A46: Education", "A48: Retraining",
     "A49: Business", "A410: Others"]
)
savings_account = st.selectbox(
    "Savings Account/Bonds",
    ["A61: < 100 DM", "A62: 100 <= … < 500 DM", "A63: 500 <= … < 1000 DM",
     "A64: >= 1000 DM", "A65: Unknown / No savings account"]
)
employment_status = st.selectbox(
    "Present Employment Since",
    ["A71: Unemployed", "A72: < 1 year", "A73: 1 <= … < 4 years", "A74: 4 <= … < 7 years", "A75: >= 7 years"]
)
personal_status_sex = st.selectbox(
    "Personal Status & Sex",
    ["A91: Male - Divorced/Separated", "A92: Female - Divorced/Separated/Married",
     "A93: Male - Single", "A94: Male - Married/Widowed", "A95: Female - Single"]
)
other_debtors = st.selectbox(
    "Other Debtors/Guarantors",
    ["A101: None", "A102: Co-applicant", "A103: Guarantor"]
)
property = st.selectbox(
    "Property",
    ["A121: Real estate", "A122: Building society savings / Life insurance",
     "A123: Car or other property", "A124: No property"]
)
other_installment_plans = st.selectbox(
    "Other Installment Plans",
    ["A141: Bank", "A142: Stores", "A143: None"]
)
housing = st.selectbox(
    "Housing",
    ["A151: Rent", "A152: Own", "A153: For free"]
)
job = st.selectbox(
    "Job",
    ["A171: Unemployed/Unskilled - Non-resident", "A172: Unskilled - Resident",
     "A173: Skilled Employee / Official", "A174: Management / Self-employed / Highly Qualified"]
)
telephone = st.selectbox(
    "Telephone",
    ["A191: None", "A192: Yes, registered under customer's name"]
)
foreign_worker = st.selectbox(
    "Foreign Worker",
    ["A201: Yes", "A202: No"]
)

# Numeric Inputs
duration = st.number_input("Duration in Months", min_value=1, value=12)
credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
installment_rate = st.number_input("Installment Rate (%) of Disposable Income", min_value=1, max_value=4, value=2)
present_residence = st.number_input("Present Residence (years)", min_value=1, max_value=4, value=2)
age = st.number_input("Age in Years", min_value=18, max_value=100, value=30)
number_of_existing_credits = st.number_input("Number of Existing Credits at This Bank", min_value=1, max_value=4, value=1)
dependents = st.number_input("Number of People Liable for Maintenance", min_value=1, max_value=2, value=1)

# Prepare data for API call
data = {
    "checking_account_status": checking_account_status.split(":")[0],
    "duration": duration,
    "credit_history": credit_history.split(":")[0],
    "purpose": purpose.split(":")[0],
    "credit_amount": credit_amount,
    "savings_account": savings_account.split(":")[0],
    "employment_status": employment_status.split(":")[0],
    "installment_rate": installment_rate,
    "personal_status_sex": personal_status_sex.split(":")[0],
    "other_debtors": other_debtors.split(":")[0],
    "present_residence": present_residence,
    "property": property.split(":")[0],
    "age": age,
    "other_installment_plans": other_installment_plans.split(":")[0],
    "housing": housing.split(":")[0],
    "number_of_existing_credits": number_of_existing_credits,
    "job": job.split(":")[0],
    "dependents": dependents,
    "telephone": telephone.split(":")[0],
    "foreign_worker": foreign_worker.split(":")[0]
}

# Convert data to DataFrame for model input
input_df = pd.DataFrame([data])

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
        st.error(f"An error occurred: {e}")
