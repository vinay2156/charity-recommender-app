import streamlit as st
import pandas as pd
import joblib
import os
import requests

# Step 1: Download .pkl model from Google Drive (only once)
FILE_ID = "1FfiLHjcA4bZkGfSvEx2OKGHoSrviFor6"  # <-- Replace with your Google Drive file ID
FILE_NAME = "charity_recommender_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(FILE_NAME):
        st.info("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        r = requests.get(url)
        with open(FILE_NAME, "wb") as f:
            f.write(r.content)
    return joblib.load(FILE_NAME)

model = load_model()

# Step 2: UI
st.title("Charity Recommender")
st.write("Enter donation details to get a recommended charity.")

amount = st.number_input("Donation Amount", min_value=1.0, value=100.0)
category = st.selectbox("Donation Category", [
    'Agriculture', 'Children', 'Community', 'Disaster Relief', 'Education',
    'Elderly Care', 'Environment', 'Food', "Girls’ Education", 'Health',
    'Health & Disaster', 'Health & Education', 'Housing', 'Medical',
    'Multiple', 'Poverty', 'Relief', 'Water'
])
location = st.selectbox("Location", [
    'Asia & Africa', 'Developing Countries', 'Global', 'India',
    'South America', 'Sub-Saharan Africa', 'USA'
])

if st.button("Get Recommendation"):
    input_df = pd.DataFrame([{"Amount": amount, "Category": category, "Location": location}])
    prediction = model.predict(input_df)[0]
    st.success(f"✅ Recommended Charity ID: {prediction}")

