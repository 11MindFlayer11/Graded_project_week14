# Visit with Us - Wellness Tourism Package Purchase Prediction (MLOps)

End-to-end MLOps pipeline that predicts whether a customer will buy the new Wellness Tourism Package.

```
tourism_project/
├── .github/workflows/pipeline.yml   # CI/CD: register data -> prepare -> train -> deploy
├── data/tourism.csv                 # raw dataset
├── model_building/
│   ├── utils.py                     # paths + Hugging Face helpers
│   ├── data_register.py             # stage 1: upload raw data to the HF dataset repo
│   ├── data_prep.py                 # stage 2: clean, split, upload train/test sets
│   └── train.py                     # stage 3: tune 5 models, track with MLflow, register best on HF Hub
├── deployment/
│   ├── app.py                       # Streamlit front end
│   ├── Dockerfile                   # container definition for the app
│   ├── requirements.txt             # app dependencies (used by Streamlit Community Cloud)
│   └── verify_deployment.py         # stage 4: deployment gate (app files + registered model)
└── requirements.txt                 # pipeline dependencies
```

## Setup
1. Create a **public** GitHub repository and put the contents of this folder at its root.
2. In GitHub: *Settings -> Secrets and variables -> Actions -> New repository secret* -> `HF_TOKEN` (a Hugging Face **write** token).
3. Push to `main`. The workflow creates the public HF dataset and model repos automatically.
4. Host the app on **Streamlit Community Cloud** (share.streamlit.io): repository = this repo, branch `main`, main file `deployment/app.py`, Python 3.12, and under *Advanced settings -> Secrets* add `MODEL_REPO_ID = "<hf-username>/tourism-wellness-model"`. It redeploys on every push.

## Run locally (no token needed)
```bash
pip install -r requirements.txt
python model_building/data_register.py && python model_building/data_prep.py && python model_building/train.py
MODEL_PATH=model_building/artifacts/model.joblib streamlit run deployment/app.py
```

Never commit tokens. The model is a decision-support tool; the decision threshold is chosen from out-of-fold
training predictions and should be revisited using real campaign costs.
