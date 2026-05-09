import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from db_connector import get_connection   # 🔹 import DB connector

# Load external CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Step 1: Load dataset
data = pd.read_csv("heart.csv")   # ✅ correct filename

X = data.drop("target", axis=1)
y = data["target"]

# Step 2: Train models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 3: Streamlit UI
st.title("❤️ Heart Disease Prediction System")

st.subheader("Patient Information Form")
# Patient ID + Name
patient_id = st.text_input("Patient ID")
patient_name = st.text_input("Patient Name")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200)
chol = st.number_input("Cholesterol", min_value=100, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar (0=Normal, 1=High)", [0, 1])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", min_value=60, max_value=250)
exang = st.selectbox("Exercise Induced Angina (0=No, 1=Yes)", [0, 1])
oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, step=0.1)
slope = st.selectbox("Slope (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

# Step 4: Prediction + Save to DB
if st.button("Predict"):
    patient_data = [[age, sex, cp, trestbps, chol, fbs, restecg,
                     thalach, exang, oldpeak, slope, ca, thal]]

    log_pred_prob = log_model.predict_proba(patient_data)[0][1]
    rf_pred_prob = rf_model.predict_proba(patient_data)[0][1]

    st.write("### 🧾 Prediction Results")
    st.write(f"Logistic Regression → No Disease: {100*(1-log_pred_prob):.2f}% | Disease: {100*log_pred_prob:.2f}%")
    st.write(f"Random Forest → No Disease: {100*(1-rf_pred_prob):.2f}% | Disease: {100*rf_pred_prob:.2f}%")

    # Save to database
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO heart_data (patient_id, patient_name,age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal,
                                log_pred_prob, rf_pred_prob)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (patient_id, patient_name,age, sex, cp, trestbps, chol, fbs, restecg,
          thalach, exang, oldpeak, slope, ca, thal,
          float(log_pred_prob), float(rf_pred_prob)))
    conn.commit()
    conn.close()

    st.markdown(
        "<p style='color:yellow; font-weight:bold;'>✅ Patient record saved to database!</p>",
        unsafe_allow_html=True
    )

# Step 5: ROC Curve
y_score = rf_model.predict