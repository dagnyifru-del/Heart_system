
from db_connector import get_connection
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

# Load external CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Step 1: Load dataset
data = pd.read_csv("heart.csv")
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
st.write("Enter patient details below:")

# Patient info
patient_id = st.text_input("🆔 Patient ID (leave blank if new)")
patient_name = st.text_input("👤 Patient Name")

# Numeric inputs
age = st.number_input("Age", 20, 100, 50)
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
thalach = st.number_input("Maximum Heart Rate Achieved", 70, 210, 150)
oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)

# Dropdowns
sex = st.selectbox("Sex", ["Female", "Male"])
cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
exang = st.selectbox("Exercise Induced Angina", [0, 1])
slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

# Encode categorical
sex_val = 0 if sex == "Female" else 1
cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
thal_map = {"Normal": 0, "Fixed Defect": 1, "Reversible Defect": 2}
cp_val = cp_map[cp]
thal_val = thal_map[thal]

# Build input dataframe
input_df = pd.DataFrame([[age, sex_val, cp_val, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal_val]],
                        columns=["age","sex","cp","trestbps","chol","fbs","restecg",
                                 "thalach","exang","oldpeak","slope","ca","thal"])

# Predictions
if st.button("Predict"):
    log_pred = log_model.predict(input_df)[0]
    rf_pred = rf_model.predict(input_df)[0]
    prediction_result = "Disease" if rf_pred == 1 else "No Disease"

    conn = get_connection()
    cursor = conn.cursor()

    # Patient ID logic
    if patient_id.strip():
        cursor.execute("SELECT * FROM heart_data WHERE id = %s", (patient_id,))
        existing = cursor.fetchone()
        if existing:
            assigned_id = patient_id
        else:
            cursor.execute("SELECT MAX(id) FROM heart_data")
            max_id = cursor.fetchone()[0]
            assigned_id = (max_id or 0) + 1
    else:
        cursor.execute("SELECT MAX(id) FROM heart_data")
        max_id = cursor.fetchone()[0]
        assigned_id = (max_id or 0) + 1

    # Insert or update record
    cursor.execute("""
        INSERT INTO heart_data (id, patient_name, age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal, prediction)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            patient_name = VALUES(patient_name),
            age = VALUES(age),
            sex = VALUES(sex),
            cp = VALUES(cp),
            trestbps = VALUES(trestbps),
            chol = VALUES(chol),
            fbs = VALUES(fbs),
            restecg = VALUES(restecg),
            thalach = VALUES(thalach),
            exang = VALUES(exang),
            oldpeak = VALUES(oldpeak),
            slope = VALUES(slope),
            ca = VALUES(ca),
            thal = VALUES(thal),
            prediction = VALUES(prediction)
    """, (assigned_id, patient_name, age, sex_val, cp_val, trestbps, chol, fbs,
          restecg, thalach, exang, oldpeak, slope, ca, thal_val, prediction_result))

    conn.commit()
    conn.close()

    # Show result to patient
    st.subheader("Results")
    st.write(f"🆔 Patient ID: {assigned_id}")
    st.write(f"👤 Patient Name: {patient_name}")
    st.write(f"Prediction: {prediction_result}")

    # ROC Curve
    st.subheader("📊 ROC Curve (Random Forest)")
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color="blue", label=f"ROC curve (area = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="red", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Heart Disease Prediction ROC Curve")
    ax.legend(loc="lower right")
    st.pyplot(fig)

# Doctor section
st.sidebar.header("👨‍⚕️ Doctor Access")
doctor_password = st.sidebar.text_input("Password", type="password")

if doctor_password == "secret123":   # replace with secure method
    search_option = st.sidebar.radio("Search by:", ["Patient ID", "Patient Name"])
    search_value = st.sidebar.text_input("Enter value")

    if st.sidebar.button("Search History"):
        conn = get_connection()
        cursor = conn.cursor()

        if search_option == "Patient ID":
            cursor.execute("SELECT * FROM heart_data WHERE id = %s", (search_value,))
        else:
            cursor.execute("SELECT * FROM heart_data WHERE patient_name LIKE %s", (f"%{search_value}%",))

        rows = cursor.fetchall()
        conn.close()

        st.subheader("📋 Search Results")
        if rows:
            df = pd.DataFrame(rows, columns=["ID","Name","Age","Sex","CP","Trestbps","Chol","FBS","RestECG",
                                             "Thalach","Exang","Oldpeak","Slope","CA","Thal","Prediction"])
            st.dataframe(df)
        else:
            st.write("No records found.")

    delete_id = st.sidebar.text_input("Enter Patient ID to Delete")
    if st.sidebar.button("Delete History"):
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM heart_data WHERE id = %s", (delete_id,))
        conn.commit()
        conn.close()
        st.sidebar.success("Patient history deleted.")
else:
    st.sidebar.info("Doctor login required to search or delete history.")
