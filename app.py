from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = Path(__file__).parent / "StudentsPerformance.csv"
CAT_COLS = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]
TARGET = "math score"


@st.cache_resource
def build_model():
    """Same pipeline as the notebook: LabelEncoder -> StandardScaler -> LinearRegression."""
    raw = pd.read_csv(DATA_PATH)
    df = raw.copy()

    encoders = {}
    for col in CAT_COLS:
        encoders[col] = LabelEncoder().fit(df[col])
        df[col] = encoders[col].transform(df[col])

    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler().fit(X_train)
    model = LinearRegression().fit(scaler.transform(X_train), y_train)

    pred = model.predict(scaler.transform(X_test))
    metrics = {
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
    }
    return model, scaler, encoders, list(X.columns), raw, metrics


def predict(model, scaler, encoders, columns, inputs):
    row = {}
    for col in columns:
        if col in encoders:
            row[col] = int(encoders[col].transform([inputs[col]])[0])
        else:
            row[col] = inputs[col]
    X = pd.DataFrame([row], columns=columns)
    score = float(model.predict(scaler.transform(X))[0])
    return min(max(score, 0.0), 100.0)


# ---------------------------- UI ----------------------------
st.set_page_config(page_title="Student Math Score Predictor", page_icon="🎓")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(160deg, #f0fdf4 0%, #dcfce7 100%); }
    .hero {
        background: linear-gradient(135deg, #15803d, #22c55e);
        color: #ffffff; padding: 28px 24px; border-radius: 18px;
        text-align: center; margin-bottom: 22px;
        box-shadow: 0 8px 20px rgba(21, 128, 61, 0.25);
    }
    .hero h1 { color: #ffffff; margin: 0; font-size: 2rem; }
    .hero p { color: #dcfce7; margin: 6px 0 0 0; }
    .result {
        background: #ffffff; border-left: 8px solid #16a34a;
        border-radius: 14px; padding: 18px 22px; margin-top: 18px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
    }
    .result .label { color: #166534; font-size: 0.95rem; }
    .result .score { color: #15803d; font-size: 2.6rem; font-weight: 700; }
    div.stButton > button, div[data-testid="stFormSubmitButton"] button {
        width: 100%; background: linear-gradient(135deg, #15803d, #22c55e);
        color: #ffffff; border: none; border-radius: 12px;
        padding: 0.6rem 1rem; font-weight: 600;
    }
    div[data-testid="stFormSubmitButton"] button:hover { filter: brightness(1.08); color: #ffffff; }
    </style>
    <div class="hero">
        <h1>🎓 Student Math Score Predictor</h1>
        <p>Linear Regression model trained on the Students Performance dataset</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model, scaler, encoders, columns, raw, metrics = build_model()

with st.form("student_form"):
    left, right = st.columns(2)
    with left:
        gender = st.selectbox("Gender", sorted(raw["gender"].unique()))
        race = st.selectbox("Race / ethnicity", sorted(raw["race/ethnicity"].unique()))
        parent = st.selectbox(
            "Parental level of education",
            sorted(raw["parental level of education"].unique()),
        )
        lunch = st.selectbox("Lunch type", sorted(raw["lunch"].unique()))
        prep = st.selectbox(
            "Test preparation course",
            sorted(raw["test preparation course"].unique()),
        )
    with right:
        reading = st.slider("Reading score", 0, 100, 70)
        writing = st.slider("Writing score", 0, 100, 70)
    submitted = st.form_submit_button("Predict math score")

if submitted:
    result = predict(
        model,
        scaler,
        encoders,
        columns,
        {
            "gender": gender,
            "race/ethnicity": race,
            "parental level of education": parent,
            "lunch": lunch,
            "test preparation course": prep,
            "reading score": reading,
            "writing score": writing,
        },
    )
    st.markdown(
        f"""
        <div class="result">
            <div class="label">Predicted math score</div>
            <div class="score">{result:.1f} / 100</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(int(round(result)))
    if result >= 75:
        st.success("Strong performance expected.")
    elif result >= 50:
        st.info("Average performance expected.")
    else:
        st.warning("This student may need extra support in math.")

with st.expander("About the model"):
    st.write(
        f"Test-set R² = **{metrics['r2']:.3f}**, "
        f"mean absolute error ≈ **{metrics['mae']:.1f}** points."
    )
    st.write(
        "Predictions are estimates from a small public dataset (1,000 students) "
        "and should not be used for real decisions about individuals."
    )
