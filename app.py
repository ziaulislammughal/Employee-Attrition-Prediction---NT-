import streamlit as st
import numpy as np
from joblib import load

# Load the saved model and scaler
model = load('models/random_forest_classifier.joblib')
scaler = load('models/scaler.joblib')  # Load the saved scaler

# Define input fields for user input
def user_input_features():
    st.sidebar.header("Employee Data")
    age = st.sidebar.slider("Age", 18, 65, 30)
    job_satisfaction = st.sidebar.slider("Job Satisfaction", 1, 4, 3)
    salary = st.sidebar.slider("Salary (in thousands)", 20, 150, 50)
    performance_rate = st.sidebar.slider("Performance Rate", 1, 5, 3)
    training_hours = st.sidebar.slider("Training Hours", 0, 80, 20)
    work_life_balance = st.sidebar.slider("Work-Life Balance", 1, 4, 2)
    tenure = st.sidebar.slider("Tenure (years)", 0, 40, 5)
    promotion_history = st.sidebar.slider("Promotion History (count)", 0, 10, 1)
    dept_rd = st.sidebar.selectbox("Department: Research & Development", [0, 1])
    dept_sales = st.sidebar.selectbox("Department: Sales", [0, 1])
    gender_male = st.sidebar.selectbox("Gender: Male", [0, 1])
    
    # Job Role selection (one-hot encoded inputs)
    job_roles = {
        "Human Resources": [1, 0, 0, 0, 0, 0, 0, 0],
        "Laboratory Technician": [0, 1, 0, 0, 0, 0, 0, 0],
        "Manager": [0, 0, 1, 0, 0, 0, 0, 0],
        "Manufacturing Director": [0, 0, 0, 1, 0, 0, 0, 0],
        "Research Director": [0, 0, 0, 0, 1, 0, 0, 0],
        "Research Scientist": [0, 0, 0, 0, 0, 1, 0, 0],
        "Sales Executive": [0, 0, 0, 0, 0, 0, 1, 0],
        "Sales Representative": [0, 0, 0, 0, 0, 0, 0, 1],
    }
    job_role = st.sidebar.selectbox("Job Role", list(job_roles.keys()))
    job_role_encoded = job_roles[job_role]

    # Combine all features into an array
    features = [age, job_satisfaction, salary, performance_rate, training_hours,
                work_life_balance, tenure, promotion_history, dept_rd, dept_sales, gender_male] + job_role_encoded
    return np.array(features).reshape(1, -1)

# Main app interface
st.title("Employee Attrition Prediction App")

# Get user input
input_data = user_input_features()

# Add a button for making predictions
if st.button("Predict"):
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make a prediction
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)

    # Display the results
    st.subheader("Prediction")
    st.write("Employee is likely to" + (" leave." if prediction[0] else " stay."))

    st.subheader("Prediction Probability")
    st.write(f"Probability of attrition: {prediction_proba[0][1]:.2f}")
    st.write(f"Probability of staying: {prediction_proba[0][0]:.2f}")
