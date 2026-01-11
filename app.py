import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("house_price_model.pkl")

st.title("🏠 House Price Prediction")

# ---- User Inputs ----
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1.0, 10.0, 2.0)
sqft_living = st.number_input("Living Area (sqft)", 300, 10000, 1500)
sqft_basement = st.number_input("Basement Area (sqft)", 0, 5000, 0)
sqft_lot = st.number_input("Lot Size (sqft)", 500, 50000, 5000)
floors = st.number_input("Floors", 1.0, 4.0, 1.0)
condition = st.slider("Condition (1–5)", 1, 5, 3)
view = st.slider("View (0–4)", 0, 4, 0)
waterfront = st.selectbox("Waterfront", [0, 1])
sqft_above = st.number_input("Sqft Above Ground", 300, 10000, 1500)

yr_built = st.number_input("Year Built", 1900, 2024, 2000)
yr_renovated = st.number_input("Year Renovated (0 if never)", 0, 2024, 0)

city = st.text_input("City", "Seattle")

# ---- Feature Engineering (MUST MATCH TRAINING) ----
CURRENT_YEAR = 2024

house_age = CURRENT_YEAR - yr_built

years_since_renovation = (
    CURRENT_YEAR - yr_renovated if yr_renovated > 0 else house_age
)

total_area = sqft_living + sqft_basement
living_lot_ratio = sqft_living / (sqft_lot + 1)

# ---- Final Input DataFrame ----
input_df = pd.DataFrame([{
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "sqft_lot": sqft_lot,
    "floors": floors,
    "condition": condition,
    "view": view,
    "waterfront": waterfront,
    "sqft_above": sqft_above,
    "house_age": house_age,
    "total_area": total_area,
    "living_lot_ratio": living_lot_ratio,
    "years_since_renovation": years_since_renovation,
    "city": city
}])

# ---- Prediction ----
if st.button("Predict Price"):
    pred_log = model.predict(input_df)
    price = np.expm1(pred_log)[0]
    st.success(f"Estimated Price: ${price:,.0f}")
