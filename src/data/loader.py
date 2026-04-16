from torch.utils.data import DataLoader
from src.data.dataset import SignLanguageDataset

class DataLoaderFactory:
    def __init__(self, train_path, test_path, batch_size=64):
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size

    def create_train_loader(self):
        train_dataset = SignLanguageDataset(self.train_path)
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def create_test_loader(self):
        test_dataset = SignLanguageDataset(self.test_path)
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
