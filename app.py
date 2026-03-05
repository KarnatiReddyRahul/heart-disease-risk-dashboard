import streamlit as st
import pandas as pd
import joblib

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="Heart Disease Risk Dashboard", layout="wide")

st.title("🏥 Heart Disease Risk Prediction System")
st.markdown("### AI-Based Clinical Risk Assessment Dashboard")

# ------------------------------
# Load Model and Features
# ------------------------------
model = joblib.load("heart_model.pkl")
model_features = joblib.load("model_features.pkl")

# ------------------------------
# Sidebar - Patient Input
# ------------------------------
st.sidebar.header("🧾 Enter Patient Details")

age = st.sidebar.number_input("Age", 18, 100, 50)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
cholesterol = st.sidebar.number_input("Cholesterol Level", 100, 400, 200)
blood_pressure = st.sidebar.number_input("Blood Pressure", 80, 200, 120)
heart_rate = st.sidebar.number_input("Heart Rate", 50, 150, 75)
exercise_hours = st.sidebar.number_input("Exercise Hours per Week", 0, 20, 3)
stress_level = st.sidebar.slider("Stress Level (1-10)", 1, 10, 5)
blood_sugar = st.sidebar.number_input("Blood Sugar", 70, 300, 100)

smoking = st.sidebar.selectbox("Smoking", ["No", "Yes", "Former"])
diabetes = st.sidebar.selectbox("Diabetes", ["No", "Yes"])
family_history = st.sidebar.selectbox("Family History", ["No", "Yes"])
obesity = st.sidebar.selectbox("Obesity", ["No", "Yes"])
alcohol = st.sidebar.selectbox("Alcohol Intake", ["None", "Moderate", "Heavy"])
angina = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
chest_pain = st.sidebar.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"]
)

# ------------------------------
# Prediction Button
# ------------------------------
if st.sidebar.button("🔍 Analyze Risk"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Cholesterol": [cholesterol],
        "Blood Pressure": [blood_pressure],
        "Heart Rate": [heart_rate],
        "Exercise Hours": [exercise_hours],
        "Stress Level": [stress_level],
        "Blood Sugar": [blood_sugar],
        "Smoking": [smoking],
        "Diabetes": [diabetes],
        "Family History": [family_history],
        "Obesity": [obesity],
        "Alcohol Intake": [alcohol],
        "Exercise Induced Angina": [angina],
        "Chest Pain Type": [chest_pain]
    })

    # Convert categorical variables
    input_data = pd.get_dummies(input_data, drop_first=True)

    # Match training features
    for col in model_features:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[model_features]

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # ------------------------------
    # Results Section
    # ------------------------------
    st.subheader("🩺 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("🔴 High Risk of Heart Disease")
        else:
            st.success("🟢 Low Risk of Heart Disease")

    with col2:
        st.metric("Risk Probability", f"{probability*100:.2f}%")

    # ------------------------------
    # Risk Progress Bar
    # ------------------------------
    st.subheader("📈 Risk Level Indicator")
    st.progress(float(probability))
    st.write(f"Risk Score: {probability*100:.2f}%")

    if probability > 0.75:
        st.error("🚨 Critical Risk Level")
    elif probability > 0.45:
        st.warning("⚠ Moderate Risk Level")
    else:
        st.success("✅ Stable Condition")

    # ------------------------------
    # Clinical Recommendations
    # ------------------------------
    st.subheader("📋 Clinical Recommendations")

    if age > 55:
        st.write("• Regular cardiac screening recommended")

    if cholesterol > 240:
        st.write("• Reduce saturated fat intake")

    if blood_pressure > 140:
        st.write("• Monitor hypertension closely")

    if smoking == "Yes":
        st.write("• Immediate smoking cessation advised")

    if exercise_hours < 2:
        st.write("• Increase physical activity")

    if stress_level > 7:
        st.write("• Stress management therapy suggested")

    if diabetes == "Yes":
        st.write("• Maintain blood glucose control")

    # ------------------------------
    # Patient Summary
    # ------------------------------
    st.subheader("🗂 Patient Clinical Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"""
        **Age:** {age}  
        **Cholesterol:** {cholesterol}  
        **Blood Pressure:** {blood_pressure}  
        **Heart Rate:** {heart_rate}  
        **Blood Sugar:** {blood_sugar}
        """)

    with col2:
        st.info(f"""
        **Smoking:** {smoking}  
        **Diabetes:** {diabetes}  
        **Family History:** {family_history}  
        **Obesity:** {obesity}  
        **Stress Level:** {stress_level}
        """)