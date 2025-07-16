import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("cheating_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---- Add custom background and styling ----
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1596495577886-d920f1fb7238?fit=crop&w=1350&q=80");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

h1 {
    color: #ffffff;
    text-align: center;
    padding: 10px;
    background-color: rgba(0, 0, 50, 0.7);
    border-radius: 10px;
}

.stButton > button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 8px;
    font-weight: bold;
}

.stMarkdown, .stSlider, .stSelectbox {
    background-color: rgba(255, 255, 255, 0.9);
    padding: 10px;
    border-radius: 8px;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ---- Banner Title ----
st.markdown("<h1>🕵️‍♀️ Online Exam Cheating Detection System</h1>", unsafe_allow_html=True)

st.subheader("Enter Student Behavior Details 👇")

# ---- User Input ----
tab_switch = st.slider("Tab Switch Count", 0, 20, 2)
time_taken = st.slider("Time Taken (seconds)", 60, 600, 300)
answer_changes = st.slider("Answer Changes", 0, 10, 1)
ip_changes = st.selectbox("IP Changed?", [0, 1])
copy_paste = st.slider("Copy-Paste Count", 0, 20, 1)
apps_open = st.selectbox("Suspicious Apps Opened?", [0, 1])

# ---- Prediction ----
if st.button("🔍 Detect Cheating"):
    features = pd.DataFrame([[tab_switch, time_taken, answer_changes, ip_changes, copy_paste, apps_open]],
                            columns=["tab_switch_count", "time_taken_sec", "answer_changes",
                                     "ip_changes", "copy_paste_count", "suspicious_apps_open"])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Cheating Detected! (Confidence: {prob:.2%})")
    else:
        st.success(f"✅ No Cheating Detected (Confidence: {1 - prob:.2%})")


