from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path

def download(dest='data/raw'):
    api = KaggleApi()
    api.authenticate()

    destp = Path(dest)
    destp.mkdir(parents=True, exist_ok=True)

    api.dataset_download_files('olistbr/brazilian-ecommerce', path=str(destp), unzip=True)
    print(f"✅ Downloaded Olist dataset to {destp.resolve()}")

if __name__ == '__main__':
    download()
