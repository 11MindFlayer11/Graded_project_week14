"""Stage 1 - Data registration: upload the raw dataset to the Hugging Face Hub."""
from utils import RAW_CSV, get_api, repo_ids

assert RAW_CSV.exists() and RAW_CSV.stat().st_size > 0, f"Missing {RAW_CSV}"
api = get_api()
if api is None:
    print(f"HF_TOKEN not set - local dataset verified at {RAW_CSV} ({RAW_CSV.stat().st_size:,} bytes). Skipping Hub upload.")
else:
    ids = repo_ids(api)
    api.create_repo(ids["dataset"], repo_type="dataset", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=str(RAW_CSV), path_in_repo="tourism.csv",
                    repo_id=ids["dataset"], repo_type="dataset",
                    commit_message="Register raw tourism dataset")
    print(f"Dataset registered: https://huggingface.co/datasets/{ids['dataset']}")
