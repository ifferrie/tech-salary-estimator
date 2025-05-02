import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('salary_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("💼 Tech Salary Estimator (Based on 2016 Data)")

# UI
job_title = st.selectbox("Job Title", ['software engineer', 'software developer', 'Data Scientist', 'Product Manager', 'Data Analyst', 'Data Engineer', 'Data Architect'])
location = st.selectbox("Location State", ['CA', 'NY', 'TX', 'Other'])
experience = st.slider("Total Experience (Years)", 0, 40, 5)
rank = st.selectbox("Job Title Rank", [1, 2, 3, 4, 'junior', 'senior'])

# Input dict
input_dict = {
    'total_experience_years': experience,
    'job_title_rank': rank,
    f'job_title_{job_title}': 1,
    f'location_state_{location}': 1,
}

# Complete input data for all model columns
input_data = pd.DataFrame([input_dict])
for col in model_columns:
    if col not in input_data:
        input_data[col] = 0
input_data = input_data[model_columns]

# Prediction
if st.button("Estimate Salary"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Annual Base Pay: ${prediction:,.2f}")