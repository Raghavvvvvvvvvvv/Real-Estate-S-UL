import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =========================================================
# Load Model
# =========================================================
model = joblib.load("house_price_model.pkl")

# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
}

.stButton>button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 8px;
    height: 45px;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
}

.prediction-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #262730;
    color: white;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# Title
# =========================================================
st.title("🏠 AI House Price Prediction System")
st.markdown("### Predict house prices using Machine Learning")

st.divider()

# =========================================================
# Sidebar
# =========================================================
st.sidebar.title("📊 About Project")

st.sidebar.info("""
This project predicts house prices using a Machine Learning Regression Model.

### Features Used
- Avg Area Income
- House Age
- Number of Rooms
- Number of Bedrooms
- Area Population
""")

# =========================================================
# Inputs
# =========================================================
st.subheader("📥 Enter House Details")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input(
        "💰 Avg. Area Income",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    house_age = st.number_input(
        "🏡 Avg. Area House Age",
        min_value=0.0,
        value=5.0,
        step=0.1
    )

    rooms = st.number_input(
        "🛏 Avg. Area Number of Rooms",
        min_value=0.0,
        value=6.0,
        step=0.1
    )

with col2:
    bedrooms = st.number_input(
        "🛌 Avg. Area Number of Bedrooms",
        min_value=0.0,
        value=3.0,
        step=0.1
    )

    population = st.number_input(
        "👨‍👩‍👧 Area Population",
        min_value=0.0,
        value=30000.0,
        step=1000.0
    )

# =========================================================
# Prediction
# =========================================================
st.divider()

if st.button("🔍 Predict House Price"):

    try:

        features = np.array([[
            income,
            house_age,
            rooms,
            bedrooms,
            population
        ]])

        prediction = model.predict(features)[0]

        st.success("✅ Prediction Successful")

        st.markdown(f"""
        <div class="prediction-box">
            Predicted House Price<br><br>
            💲 {prediction:,.2f}
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# =========================================================
# Feature Table
# =========================================================
st.divider()

feature_data = pd.DataFrame({
    "Feature": [
        "Avg. Area Income",
        "Avg. Area House Age",
        "Avg. Area Number of Rooms",
        "Avg. Area Number of Bedrooms",
        "Area Population"
    ],
    "Description": [
        "Average income of people in area",
        "Average age of houses",
        "Average number of rooms",
        "Average bedrooms in houses",
        "Population of area"
    ]
})

st.dataframe(
    feature_data,
    use_container_width=True,
    hide_index=True
)

# =========================================================
# Footer
# =========================================================
st.divider()

st.markdown("""
<center>
Made with ❤️ using Streamlit + Machine Learning
</center>
""", unsafe_allow_html=True)