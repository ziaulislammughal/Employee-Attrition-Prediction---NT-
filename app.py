import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load pre-trained model
model = load('random_forest_classifier.joblib')  # Adjust the path if needed

# Set up Streamlit interface
st.title("Employee Attrition Prediction App")
st.subheader("Predict if an employee will leave the company based on input data.")

# Sidebar inputs for user data
st.sidebar.header("Employee Information Input")

# Define input fields for user
def get_user_input():
    Department = st.sidebar.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    EmployeeID = st.sidebar.text_input("EmployeeID", "")
    Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    JobRole = st.sidebar.selectbox("JobRole", ["Manager", "Laboratory Technician", "Sales Executive", "Research Scientist", 
                                               "Human Resources", "Manufacturing Director", "Healthcare Representative", 
                                               "Research Director", "Sales Representative"])
    JobSatisfaction = st.sidebar.slider("JobSatisfaction (1-4)", 1, 4)
    Salary = st.sidebar.number_input("Salary", min_value=0)
    PerformanceRate = st.sidebar.slider("PerformanceRate (1-4)", 1, 4)
    TrainingHours = st.sidebar.number_input("TrainingHours", min_value=0)
    WorkLifeBalance = st.sidebar.slider("WorkLifeBalance (1-4)", 1, 4)
    Tenure = st.sidebar.number_input("Tenure", min_value=0)
    PromotionHistory = st.sidebar.number_input("PromotionHistory", min_value=0)

    # Convert categorical features to numeric (use 1 for Male, 0 for Female as an example)
    Gender = 1 if Gender == "Male" else 0

    # Assemble input data into DataFrame for prediction
    data = pd.DataFrame({
        "Department": [Department],
        "EmployeeID": [float(EmployeeID) if EmployeeID.isnumeric() else 0],  # Convert to float or use 0 as default
        "Gender": [Gender],
        "JobRole": [JobRole],
        "JobSatisfaction": [JobSatisfaction],
        "Salary": [Salary],
        "PerformanceRate": [PerformanceRate],
        "TrainingHours": [TrainingHours],
        "WorkLifeBalance": [WorkLifeBalance],
        "Tenure": [Tenure],
        "PromotionHistory": [PromotionHistory]
    })

    # One-hot encode categorical columns as needed to match model input format
    data = pd.get_dummies(data)
    return data

user_data = get_user_input()

# Prediction button
if st.button("Predict"):
    # Ensure the data matches the model's required features
    try:
        prediction = model.predict(user_data)[0]
        result = "Leave" if prediction == 1 else "Stay"
        st.write(f"Prediction: The employee is likely to **{result}**.")
    except Exception as e:
        st.write("Error with input data. Please check input fields:", e)
