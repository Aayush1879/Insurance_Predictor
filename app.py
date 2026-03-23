import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page config — must be first Streamlit call ───────────────────────────────
st.set_page_config(
    page_title="Insurance Claim Predictor",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f1724 0%, #1a2640 50%, #0f1f35 100%);
    min-height: 100vh;
}

/* ── Header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,179,237,0.12);
    border: 1px solid rgba(99,179,237,0.3);
    color: #63b3ed;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #f0f6ff;
    line-height: 1.15;
    margin: 0 0 0.5rem;
}
.hero-title span {
    color: #63b3ed;
    font-style: italic;
}
.hero-sub {
    color: #8ba4c0;
    font-size: 1rem;
    font-weight: 300;
    max-width: 420px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Card ── */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 2rem 2rem 1.5rem;
    margin: 1.5rem 0;
    backdrop-filter: blur(12px);
}
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(99,179,237,0.2);
}

/* ── Inputs ── */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    color: #a8c0d6 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    margin-bottom: 0.2rem !important;
}
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8f0f7 !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: rgba(99,179,237,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.1) !important;
}

/* ── Button ── */
div[data-testid="stFormSubmitButton"] button {
    width: 100%;
    background: linear-gradient(135deg, #2b6cb0, #3182ce) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    margin-top: 0.5rem !important;
}
div[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(135deg, #2c5282, #2b6cb0) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(49,130,206,0.4) !important;
}

/* ── Result box ── */
.result-box {
    background: linear-gradient(135deg, rgba(49,130,206,0.15), rgba(99,179,237,0.08));
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
    animation: fadeUp 0.5s ease forwards;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 0.5rem;
}
.result-amount {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #f0f6ff;
    line-height: 1;
}
.result-note {
    font-size: 0.8rem;
    color: #7a98b5;
    margin-top: 0.75rem;
}

/* ── Risk pill ── */
.risk-pill {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    margin-top: 0.8rem;
}
.risk-low  { background: rgba(72,187,120,0.15); color: #68d391; border: 1px solid rgba(72,187,120,0.3); }
.risk-mid  { background: rgba(237,187,70,0.15);  color: #f6c90e; border: 1px solid rgba(237,187,70,0.3); }
.risk-high { background: rgba(252,129,74,0.15);  color: #fc8149; border: 1px solid rgba(252,129,74,0.3); }

/* ── Metric strip ── */
.metric-strip {
    display: flex;
    gap: 1rem;
    margin-top: 1.2rem;
}
.metric-item {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 0.9rem;
    text-align: center;
}
.metric-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #e8f0f7;
}
.metric-lbl {
    font-size: 0.7rem;
    color: #6a88a0;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #3d5a73;
    font-size: 0.75rem;
    padding: 2rem 0 1rem;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.pkl")
    gender_le = joblib.load("gender_label_encoder.pkl")
    diabetic_le = joblib.load("diabetic_label_encoder.pkl")
    smoker_le = joblib.load("smoker_label_encoder.pkl")
    model = joblib.load("best_model.pkl")
    return scaler, gender_le, diabetic_le, smoker_le, model


scaler, gender_le, diabetic_le, smoker_le, model = load_artifacts()

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
    <div class="hero-badge">AI-Powered Estimator</div>
    <h1 class="hero-title">Insurance <span>Claim</span> Predictor</h1>
    <p class="hero-sub">Enter your health profile below to get an instant AI estimate of your insurance claim amount.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Form ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">👤 Personal Info</div>', unsafe_allow_html=True)

with st.form("claim_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        gender = st.selectbox("Gender", options=gender_le.classes_)
        children = st.number_input(
            "Number of Children", min_value=0, max_value=10, value=0
        )
    with col2:
        bmi = st.number_input(
            "BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1
        )
        bloodpressure = st.number_input(
            "Blood Pressure (mmHg)", min_value=40.0, max_value=200.0, value=120.0
        )

    st.markdown(
        '<div class="section-label" style="margin-top:1.2rem">🩺 Health Status</div>',
        unsafe_allow_html=True,
    )
    col3, col4 = st.columns(2)
    with col3:
        diabetic = st.selectbox("Diabetic", options=diabetic_le.classes_)
    with col4:
        smoker = st.selectbox("Smoker", options=smoker_le.classes_)

    submitted = st.form_submit_button("⚡ Predict Claim Amount")

st.markdown("</div>", unsafe_allow_html=True)


# ── BMI helper ────────────────────────────────────────────────────────────────
def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    input_data = pd.DataFrame(
        {
            "age": [age],
            "gender": [gender],
            "bmi": [bmi],
            "bloodpressure": [bloodpressure],
            "children": [children],
            "diabetic": [diabetic],
            "smoker": [smoker],
        }
    )

    input_data["gender"] = gender_le.transform(input_data["gender"])
    input_data["diabetic"] = diabetic_le.transform(input_data["diabetic"])
    input_data["smoker"] = smoker_le.transform(input_data["smoker"])

    num_cols = ["age", "bmi", "bloodpressure", "children"]
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    prediction = model.predict(input_data)[0]
    prediction = max(0, prediction)  # clamp negatives

    # Risk tier
    if prediction < 10_000:
        risk_cls, risk_lbl = "risk-low", "🟢 Low Risk"
    elif prediction < 25_000:
        risk_cls, risk_lbl = "risk-mid", "🟡 Moderate Risk"
    else:
        risk_cls, risk_lbl = "risk-high", "🔴 High Risk"

    st.markdown(
        f"""
    <div class="result-box">
        <div class="result-label">Estimated Claim Amount</div>
        <div class="result-amount">${prediction:,.0f}</div>
        <div><span class="risk-pill {risk_cls}">{risk_lbl}</span></div>
        <div class="result-note">Based on your health profile · Results are estimates only</div>
    </div>

    <div class="metric-strip">
        <div class="metric-item">
            <div class="metric-val">{age}</div>
            <div class="metric-lbl">Age</div>
        </div>
        <div class="metric-item">
            <div class="metric-val">{bmi:.1f}</div>
            <div class="metric-lbl">BMI · {bmi_category(bmi)}</div>
        </div>
        <div class="metric-item">
            <div class="metric-val">{bloodpressure:.0f}</div>
            <div class="metric-lbl">Blood Pressure</div>
        </div>
        <div class="metric-item">
            <div class="metric-val">{children}</div>
            <div class="metric-lbl">Children</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footer">Insurance Claim Predictor · Powered by Machine Learning</div>',
    unsafe_allow_html=True,
)
