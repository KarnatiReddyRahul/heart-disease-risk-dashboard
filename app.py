import streamlit as st
import pandas as pd
import joblib
from groq import Groq

# -------------------------------------------------------
# Page Config
# -------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Dashboard",
    layout="wide",
    page_icon="🫀"
)

st.title("🫀 Heart Disease Risk Prediction System")
st.markdown("### AI-Based Clinical Risk Assessment Dashboard")

# -------------------------------------------------------
# Load Model and Features
# -------------------------------------------------------
model = joblib.load("heart_model.pkl")
model_features = joblib.load("model_features.pkl")

# -------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_risk_score" not in st.session_state:
    st.session_state.last_risk_score = None

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "patient_data" not in st.session_state:
    st.session_state.patient_data = {}

# -------------------------------------------------------
# Sidebar - Patient Input
# -------------------------------------------------------
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

# -------------------------------------------------------
# Tabs — only 2 tabs now
# -------------------------------------------------------
tab1, tab2 = st.tabs(["🩺 Risk Prediction", "🤖 AI Health Assistant"])

# ===============================================================
# TAB 1 — Risk Prediction
# ===============================================================
with tab1:
    if st.sidebar.button("🔍 Analyze Risk"):

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

        input_data = pd.get_dummies(input_data, drop_first=True)

        for col in model_features:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[model_features]

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # Save to session state so chatbot can reference it
        st.session_state.last_risk_score = probability
        st.session_state.last_prediction = prediction
        st.session_state.patient_data = {
            "age": age, "gender": gender, "cholesterol": cholesterol,
            "blood_pressure": blood_pressure, "heart_rate": heart_rate,
            "exercise_hours": exercise_hours, "stress_level": stress_level,
            "blood_sugar": blood_sugar, "smoking": smoking,
            "diabetes": diabetes, "family_history": family_history,
            "obesity": obesity, "alcohol": alcohol
        }

        # Results
        st.subheader("🩺 Prediction Result")
        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("🔴 High Risk of Heart Disease")
            else:
                st.success("🟢 Low Risk of Heart Disease")

        with col2:
            st.metric("Risk Probability", f"{probability*100:.2f}%")

        st.subheader("📈 Risk Level Indicator")
        st.progress(float(probability))
        st.write(f"Risk Score: {probability*100:.2f}%")

        if probability > 0.75:
            st.error("🚨 Critical Risk Level")
        elif probability > 0.45:
            st.warning("⚠ Moderate Risk Level")
        else:
            st.success("✅ Stable Condition")

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

    else:
        st.info("👈 Fill in the patient details in the sidebar and click **Analyze Risk** to see the prediction.")

# ===============================================================
# TAB 2 — AI Health Assistant (Google Gemini - new SDK)
# ===============================================================
with tab2:
    st.subheader("🤖 AI Heart Health Assistant")
    st.markdown("Ask me anything about heart disease, your risk result, medications, lifestyle tips, and more.")

    # Show risk context if prediction was done
    if st.session_state.last_risk_score is not None:
        risk_pct = st.session_state.last_risk_score * 100
        label = "High Risk" if st.session_state.last_prediction == 1 else "Low Risk"
        st.info(
            f"ℹ️ Your latest prediction: **{label}** ({risk_pct:.1f}% risk score). "
            "The assistant is aware of this and can answer based on your results."
        )

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask a heart health question...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build patient context for system prompt
        patient_context = ""
        if st.session_state.last_risk_score is not None:
            pd_data = st.session_state.patient_data
            patient_context = f"""
The patient has been assessed with the following details:
- Risk Score: {st.session_state.last_risk_score * 100:.1f}%
- Prediction: {"High Risk" if st.session_state.last_prediction == 1 else "Low Risk"}
- Age: {pd_data.get('age')}, Gender: {pd_data.get('gender')}
- Cholesterol: {pd_data.get('cholesterol')}, Blood Pressure: {pd_data.get('blood_pressure')}
- Heart Rate: {pd_data.get('heart_rate')}, Blood Sugar: {pd_data.get('blood_sugar')}
- Smoking: {pd_data.get('smoking')}, Diabetes: {pd_data.get('diabetes')}
- Stress Level: {pd_data.get('stress_level')}/10, Exercise: {pd_data.get('exercise_hours')} hrs/week
- Family History: {pd_data.get('family_history')}, Obesity: {pd_data.get('obesity')}
Use this to give personalized answers when relevant.
"""

        system_prompt = f"""You are a compassionate, knowledgeable AI assistant specialized in cardiovascular health and heart disease prevention.
You help patients understand their heart health risks, explain medical terms in simple language,
suggest healthy lifestyle changes, and answer questions about heart conditions, medications, and when to see a doctor.

Always recommend consulting a real doctor for serious concerns. Be warm, clear, and supportive.
Never diagnose or prescribe — only educate and guide.

{patient_context}"""

        # Call Gemini API using NEW official SDK
        try:
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])

            messages = [{"role": "system", "content": system_prompt}]
            for msg in st.session_state.chat_history[:-1]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": user_input})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=messages,
                        max_tokens=1000
                    )
                    reply = response.choices[0].message.content
                    st.markdown(reply)

            st.session_state.chat_history.append({"role": "assistant", "content": reply})

        except Exception as e:
            st.error(f"❌ AI Assistant error: {e}")
            st.info("Check that GROQ_API_KEY is correctly set in `.streamlit/secrets.toml`")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()