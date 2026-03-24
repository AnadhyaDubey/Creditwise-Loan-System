import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
ohe = pickle.load(open("ohe.pkl", "rb"))

st.title("CreditWise — Loan Approval Predictor")

credit_score = st.slider("Credit Score", 300, 900, 650)
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Co-applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
dti_ratio = st.number_input("DTI Ratio", min_value=0.0)
savings = st.number_input("Savings", min_value=0)
loan_term = st.number_input("Loan Term (months)", min_value=0)
age = st.number_input("Age", min_value=18)
education = st.selectbox("Education Level", ["Graduate", "Not Graduate", "Postgraduate"])
employment = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])
marital = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
purpose = st.selectbox("Loan Purpose", ["Home", "Education", "Business", "Personal"])
area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])
gender = st.selectbox("Gender", ["Male", "Female"])
employer_cat = st.selectbox("Employer Category", ["Government", "Private", "NGO"])

if st.button("Predict"):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    cat_cols = ['Employment_Status', 'Marital_Status', 'Loan_Purpose',
                'Property_Area', 'Gender', 'Employer_Category']

    input_dict = {
        'Credit_Score': credit_score,
        'Applicant_Income': applicant_income,
        'Coapplicant_Income': coapplicant_income,
        'Loan_Amount': loan_amount,
        'DTI_Ratio': dti_ratio,
        'Savings': savings,
        'Loan_Term': loan_term,
        'Age': age,
        'Education_Level': education,
        'Employment_Status': employment,
        'Marital_Status': marital,
        'Loan_Purpose': purpose,
        'Property_Area': area,
        'Gender': gender,
        'Employer_Category': employer_cat
    }

    input_df = pd.DataFrame([input_dict])

    edu_map = {"Graduate": 1, "Not Graduate": 0, "Postgraduate": 2}
    input_df["Education_Level"] = input_df["Education_Level"].map(edu_map)

    encoded_cats = ohe.transform(input_df[cat_cols])
    encoded_df = pd.DataFrame(encoded_cats, columns=ohe.get_feature_names_out(cat_cols))
    input_df = pd.concat([input_df.drop(columns=cat_cols), encoded_df], axis=1)

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("Loan Approved ✓")
    else:
        st.error("Loan Not Approved ✗")
