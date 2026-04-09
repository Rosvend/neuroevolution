import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

class DatasetDownloader:
    def __init__(self, dataset_name="datamunge/sign-language-mnist", download_path="datamunge/sign-language-mnist"):
        self.dataset_name = dataset_name
        self.download_path = download_path
        self.api = KaggleApi()
        
    def download_and_extract(self):
        if not os.path.exists(f"{self.download_path}/sign_mnist_train.csv"):
            self.api.authenticate()
            os.makedirs(self.download_path, exist_ok=True)
            self.api.dataset_download_files(self.dataset_name, path=self.download_path, unzip=True)
