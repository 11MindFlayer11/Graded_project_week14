"""Streamlit front end for the Wellness Tourism Package purchase model."""
import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="Wellness Tourism Lead Predictor", page_icon="✈️", layout="centered")


@st.cache_resource(show_spinner="Loading the registered model from the Hugging Face Hub...")
def load_bundle():
    """Local file (for testing) first, otherwise download the registered model from the Hub."""
    local = os.getenv("MODEL_PATH")
    if local and Path(local).exists():
        return joblib.load(local)
    repo_id = os.getenv("MODEL_REPO_ID")
    if not repo_id:
        try:
            repo_id = st.secrets["MODEL_REPO_ID"]   # Streamlit Community Cloud -> Advanced settings -> Secrets
        except Exception:  # noqa: BLE001
            repo_id = None
    cfg = BASE / "config.json"
    if not repo_id and cfg.exists():
        repo_id = json.loads(cfg.read_text())["model_repo_id"]
    if not repo_id:
        raise RuntimeError("No model source configured: add MODEL_REPO_ID = \"<hf-username>/tourism-wellness-model\" to the app secrets.")
    from huggingface_hub import hf_hub_download
    return joblib.load(hf_hub_download(repo_id=repo_id, filename="model.joblib"))


st.title("Wellness Tourism Package - Lead Predictor")
st.write("Estimate how likely a customer is to purchase the Wellness Tourism Package *before* contacting them, "
         "so outreach can be prioritised. This is decision support, not a guarantee of behaviour.")

try:
    bundle = load_bundle()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load the model: {exc}")
    st.stop()
model, threshold = bundle["pipeline"], bundle["threshold"]

with st.form("customer_form"):
    st.subheader("Customer details")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 18, 100, 36)
        contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
        city = st.selectbox("City tier (1 = most developed)", [1, 2, 3])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        visitors = st.number_input("Number of persons visiting", 1, 10, 3)
        stars = st.selectbox("Preferred property star rating", [3, 4, 5])
        marital = st.selectbox("Marital status", ["Married", "Single", "Unmarried", "Divorced"])
        trips = st.number_input("Trips per year", 0, 50, 3)
        passport = st.selectbox("Holds a valid passport?", [0, 1], format_func=lambda v: "Yes" if v else "No")
    with c2:
        car = st.selectbox("Owns a car?", [1, 0], format_func=lambda v: "Yes" if v else "No")
        children = st.number_input("Children below age 5 visiting", 0, 10, 1)
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        income = st.number_input("Monthly income", 1000, 500000, 22000, step=1000)
        satisfaction = st.selectbox("Pitch satisfaction score", [1, 2, 3, 4, 5], index=2)
        product = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        followups = st.number_input("Number of follow-ups", 0, 20, 4)
        duration = st.number_input("Pitch duration (minutes)", 1.0, 200.0, 14.0)
    submitted = st.form_submit_button("Predict purchase likelihood")

if submitted:
    row = pd.DataFrame([{
        "Age": age, "TypeofContact": contact, "CityTier": city, "DurationOfPitch": duration,
        "Occupation": occupation, "Gender": gender, "NumberOfPersonVisiting": visitors,
        "NumberOfFollowups": followups, "ProductPitched": product, "PreferredPropertyStar": stars,
        "MaritalStatus": marital, "NumberOfTrips": trips, "Passport": passport,
        "PitchSatisfactionScore": satisfaction, "OwnCar": car, "NumberOfChildrenVisiting": children,
        "Designation": designation, "MonthlyIncome": income,
    }])[bundle["features"]]
    prob = float(model.predict_proba(row)[0, 1])
    st.metric("Estimated purchase probability", f"{prob:.1%}")
    if prob >= threshold:
        st.success(f"Likely buyer (probability is above the model's decision threshold of {threshold:.2f}). "
                   "Prioritise for follow-up.")
    else:
        st.info(f"Less likely buyer (below the decision threshold of {threshold:.2f}). "
                "Consider lower-priority outreach.")
