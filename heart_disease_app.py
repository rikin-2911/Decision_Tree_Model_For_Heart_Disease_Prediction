import streamlit as st
import numpy as np
import pickle

# Load the trained model
def load_model():
    with open('heart_disease_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Customize Page Layout
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# Background Style
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://www.linktoyourbackgroundimage.com/image.jpg");
    background-size: cover;
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.8);
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Title and Description
st.title("❤️ Heart Disease Prediction")
st.markdown("**Provide your health details to check the risk of heart disease.**")

# Sidebar for Input Features
st.sidebar.header("🔧 Input Features")
st.sidebar.markdown("Adjust the sliders and options below:")

# Interactive User Inputs
age = st.sidebar.slider("Age", min_value=1, max_value=120, value=50, help="Age in years")
sex = st.sidebar.radio("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.sidebar.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], help="0: Typical Angina, 1: Atypical Angina, 2: Non-Anginal Pain, 3: Asymptomatic")
trestbps = st.sidebar.slider("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120, help="In mm Hg")
chol = st.sidebar.slider("Serum Cholesterol (chol)", min_value=100, max_value=600, value=200, help="In mg/dl")
fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
restecg = st.sidebar.selectbox("Resting ECG (restecg)", options=[0, 1, 2], help="0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy")
thalach = st.sidebar.slider("Max Heart Rate Achieved (thalach)", min_value=50, max_value=250, value=150, help="Maximum Heart Rate Achieved")
exang = st.sidebar.radio("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.sidebar.slider("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="ST Depression Induced by Exercise Relative to Rest")
slope = st.sidebar.selectbox("Slope of Peak Exercise ST (slope)", options=[0, 1, 2], help="0: Upsloping, 1: Flat, 2: Downsloping")
ca = st.sidebar.slider("Number of Major Vessels (ca)", min_value=0, max_value=4, value=0, help="Number of Major Vessels Colored by Fluoroscopy")
thal = st.sidebar.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3], help="0: Normal, 1: Fixed Defect, 2: Reversible Defect, 3: Unknown")

# Convert inputs to numpy array
input_features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

# Predict
if st.button("🔍 Predict"):
    prediction = model.predict(input_features)
    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected! Please consult a doctor.")
    else:
        st.success("✅ No Heart Disease Detected. Keep maintaining a healthy lifestyle!")

    # Explanation
    st.markdown("---")
    st.markdown("🔎 **Model Explanation:**")
    st.markdown("""
        - This prediction is made by a Machine Learning model.
        - It is based on the input features you provided.
        - This is not a medical diagnosis. Please consult a healthcare professional for accurate health assessments.
    """)

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ by Rikin Pithadia.")
