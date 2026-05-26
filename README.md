# ❤️ Heart Disease Risk Prediction Dashboard

## 📌 Project Overview
This project predicts the risk of heart disease using Machine Learning based on patient health data such as age, blood pressure, cholesterol level, and lifestyle factors. It also includes an **AI-powered chatbot** that answers heart health questions in real time.

## 🚀 Technologies Used
- Python
- Pandas
- Scikit-learn (Random Forest Classifier)
- Streamlit
- Groq API (LLaMA 3.1 - AI Chatbot)

## 📊 Features
- ✅ User-friendly clinical dashboard
- ✅ Predicts heart disease risk with probability score
- ✅ Risk level indicator (Critical / Moderate / Stable)
- ✅ Personalized clinical recommendations
- ✅ AI Health Assistant chatbot (powered by Groq + LLaMA 3.1)
- ✅ Chatbot is aware of the patient's risk score and health data
- ✅ Real-time prediction from user inputs

## 🧠 Machine Learning Model
- **Model Used:** Random Forest Classifier
- **Input Features:** Age, Gender, Cholesterol, Blood Pressure, Heart Rate, Blood Sugar, Exercise Hours, Stress Level, Smoking, Diabetes, Family History, Obesity, Alcohol Intake, Exercise Induced Angina, Chest Pain Type

## 🤖 AI Chatbot
- Powered by **Groq API** with **LLaMA 3.1 8B** model
- Answers questions about heart health, medications, lifestyle tips
- Personalized responses based on the patient's risk assessment
- Free to use with no billing required

## ⚙️ How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/KarnatiReddyRahul/heart-disease-risk-dashboard.git
cd heart-disease-risk-dashboard
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API Key
Create a file at `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your-groq-api-key-here"
```
Get your free Groq API key at: https://console.groq.com

### 5. Run the app
```bash
streamlit run app.py
```

## 📁 Project Structure
```
heart-disease-risk-dashboard/
├── app.py                  # Main Streamlit application
├── heart_model.pkl         # Trained ML model
├── model_features.pkl      # Model feature names
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore file
├── .streamlit/
│   └── secrets.toml       # API keys (do NOT push to GitHub)
└── README.md              # Project documentation
```

## 📦 Requirements
```
streamlit
pandas
numpy
scikit-learn
joblib
groq
```

## 🌐 Live Demo
Try the live dashboard here:
https://heart-disease-predictor26.streamlit.app/

## 📈 Future Improvements
- Add more patient datasets for better accuracy
- Add visualizations and charts for risk factors
- Include medication database integration
- Multi-language support

## ⚠️ Disclaimer
This application is for **educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.

## 👨‍💻 Developer
**Karnati Reddy Rahul**  
GitHub: https://github.com/KarnatiReddyRahul
