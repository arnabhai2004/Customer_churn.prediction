import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Churn Prediction App")

st.divider()

st.write("Please enter the values and hit the prediction button for getting a prediction.")

st.divider()

age = st.number_input("Enter age", min_value=10, max_value=100, value=30)

gender = st.selectbox("Select gender", options=["Male", "Female"])

tenure = st.number_input("Enter tenure (in months)", min_value=0, max_value=130, value=5)

monthlycharge = st.number_input("Enter monthly charge", min_value=0.0, max_value=1000.0, value=50.0)

st.divider()

predictbutton = st.button("Predict")   # ← এই line টা যোগ হয়েছে

if predictbutton:
    gender_selected = 1 if gender == "Female" else 0
    x = np.array([[age, gender_selected, tenure, monthlycharge]])
    x_scaled = scaler.transform(x)

    prediction = model.predict(x_scaled)[0]
    predicted = 'Yes' if prediction == 1 else 'No'

    st.balloons()  # ← এই line টা যোগ হয়েছে

    st.write(f"The predicted class is: {predicted}")

else:
    st.write("Please click the prediction button to get the prediction.")