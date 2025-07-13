import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------- Page Config -------------------
st.set_page_config(page_title="🚗 Car Price Predictor", layout="centered")
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .title {
        text-align: center;
        padding: 0.8rem;
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
        border-radius: 12px;
    }
    .card {
        padding: 1rem;
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 1rem;
    }
    .btn {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        padding: 0.6rem 1.2rem;
        border-radius: 10px;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 class='title'>🚗 Car Price Predictor</h2>", unsafe_allow_html=True)

# ------------------- Load Data & Model -------------------
@st.cache_data
def load_data():
    df = pd.read_csv('quikr_car.csv')
    df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)
    df = df[df['price'].str.lower() != 'ask for price']
    df.dropna(subset=['company', 'fuel_type', 'year', 'kms_driven'], inplace=True)
    df['company'] = df['company'].astype(str).str.strip()
    df['fuel_type'] = df['fuel_type'].astype(str).str.strip()
    return df

df = load_data()

model = pickle.load(open('linearregressionmodel.pkl', 'rb'))

# ------------------- Input Section -------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("🔧 Enter Car Details:")

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("🏢 Car Company", sorted(df['company'].unique()))

with col2:
    fuel_type = st.selectbox("⛽ Fuel Type", sorted(df['fuel_type'].dropna().unique()))

col3, col4 = st.columns(2)

with col3:
    kms_driven = st.number_input("🧭 Kilometers Driven", min_value=0, step=1000, value=20000)

with col4:
    car_age = st.slider("📅 Car Age (in years)", min_value=0, max_value=30, value=5)

st.markdown("</div>", unsafe_allow_html=True)

# ------------------- Prepare Model Input -------------------
input_dict = {
    'kms_driven': [kms_driven],
    'car_age': [car_age]
}

# Manual one-hot encode to match model features
for c in model.feature_names_in_:
    if c.startswith('company_'):
        input_dict[c] = [1 if c == f'company_{company}' else 0]
    elif c.startswith('fuel_type_'):
        input_dict[c] = [1 if c == f'fuel_type_{fuel_type}' else 0]

# Fill in any missing columns
for col in model.feature_names_in_:
    if col not in input_dict:
        input_dict[col] = [0]

input_df = pd.DataFrame(input_dict)

# ------------------- Prediction Section -------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📈 Predict Car Price")

if st.button("🔍 Estimate Price"):
    predicted_price = model.predict(input_df)[0]
    st.success(f"💰 Estimated Resale Price: ₹ {int(predicted_price):,}")
else:
    st.info("Click the button above to see estimated car price.")

st.markdown("</div>", unsafe_allow_html=True)
