import torch
import pandas as pd
from torch.utils.data import Dataset

class SignLanguageDataset(Dataset):
    def __init__(self, csv_file_path, transform=None):
        data_frame = pd.read_csv(csv_file_path)
        labels_raw = data_frame.iloc[:, 0].values
        self.labels = torch.tensor(
            [label if label < 9 else label - 1 for label in labels_raw], 
            dtype=torch.long
        )
        self.features = torch.tensor(data_frame.iloc[:, 1:].values / 255.0, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        feature_sample = self.features[index]
        label_sample = self.labels[index]

        if self.transform:
            feature_sample = self.transform(feature_sample)

        return feature_sample, label_sample
