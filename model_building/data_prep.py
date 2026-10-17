"""Stage 2 - Data preparation: clean, split and publish train/test sets."""
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import RAW_CSV, PROCESSED_DIR, SPLIT_FILES, get_api, repo_ids

api = get_api()
if api is None:
    print("HF_TOKEN not set - reading the local dataset.")
    df = pd.read_csv(RAW_CSV)
else:
    from huggingface_hub import hf_hub_download
    ids = repo_ids(api)
    df = pd.read_csv(hf_hub_download(ids["dataset"], "tourism.csv", repo_type="dataset", token=api.token))
print("Raw shape:", df.shape)

# ---- cleaning ---------------------------------------------------------------
df = df.drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns])  # index / identifier
for col in df.select_dtypes(include=["object", "string"]):
    df[col] = df[col].str.strip()
fixed = int((df["Gender"] == "Fe Male").sum())
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})  # typo in source data
print(f"Fixed {fixed} 'Fe Male' -> 'Female' values")
before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} duplicate rows")
print("Missing values left:", int(df.isna().sum().sum()), "(any future gaps are imputed inside the model pipeline)")

# ---- split ------------------------------------------------------------------
X, y = df.drop(columns=["ProdTaken"]), df["ProdTaken"].astype(int)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
Xtrain.to_csv(PROCESSED_DIR / "Xtrain.csv", index=False)
Xtest.to_csv(PROCESSED_DIR / "Xtest.csv", index=False)
ytrain.to_csv(PROCESSED_DIR / "ytrain.csv", index=False)
ytest.to_csv(PROCESSED_DIR / "ytest.csv", index=False)
print(f"Train: {Xtrain.shape}  Test: {Xtest.shape}  | positive rate train={ytrain.mean():.3f} test={ytest.mean():.3f}")

if api is not None:
    for name in SPLIT_FILES:
        api.upload_file(path_or_fileobj=str(PROCESSED_DIR / name), path_in_repo=f"processed/{name}",
                        repo_id=ids["dataset"], repo_type="dataset", commit_message=f"Add {name}")
    print(f"Processed splits uploaded to https://huggingface.co/datasets/{ids['dataset']}")
