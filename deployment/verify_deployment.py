"""Stage 4 - Deployment gate, run by the last GitHub Actions job.

The Streamlit app is hosted on Streamlit Community Cloud, which redeploys automatically whenever
`main` changes. Before that happens this script checks that the deployment is healthy:
  1. the app files exist, and
  2. the registered model can be downloaded from the Hugging Face Hub, loaded and used for a prediction.
Set MODEL_PATH to test against a local model file instead of the Hub.
"""
import os
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
missing = [f for f in ["app.py", "requirements.txt", "Dockerfile"] if not (HERE / f).exists()]
assert not missing, f"Missing deployment files: {missing}"
print("Deployment files present: app.py, requirements.txt, Dockerfile")

local = os.getenv("MODEL_PATH")
if local:
    model_path, source = local, "local file"
else:
    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ["HF_TOKEN"]
    user = os.getenv("HF_USERNAME") or HfApi(token=token).whoami()["name"]
    repo = os.getenv("HF_MODEL_REPO", f"{user}/tourism-wellness-model")
    model_path, source = hf_hub_download(repo, "model.joblib", token=token), f"Hugging Face Hub ({repo})"

bundle = joblib.load(model_path)
sample = pd.DataFrame([{
    "Age": 30, "TypeofContact": "Self Enquiry", "CityTier": 1, "DurationOfPitch": 14.0, "Occupation": "Salaried",
    "Gender": "Male", "NumberOfPersonVisiting": 3, "NumberOfFollowups": 4, "ProductPitched": "Basic",
    "PreferredPropertyStar": 3, "MaritalStatus": "Single", "NumberOfTrips": 3, "Passport": 1,
    "PitchSatisfactionScore": 3, "OwnCar": 1, "NumberOfChildrenVisiting": 1, "Designation": "Executive",
    "MonthlyIncome": 22000}])[bundle["features"]]
prob = float(bundle["pipeline"].predict_proba(sample)[0, 1])
assert 0.0 <= prob <= 1.0
print(f"Model loaded from {source}: {bundle['model_name']}, decision threshold {bundle['threshold']}")
print(f"Smoke-test prediction OK (purchase probability {prob:.1%}).")
print("Deployment checks passed - Streamlit Community Cloud redeploys the app automatically from main.")
