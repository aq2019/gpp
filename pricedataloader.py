from torch.utils.data import Dataset

class TorchDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        sample_x = self.x[idx]
        sample_y = self.y[idx]
        if self.transform is not None:
            sample_x, sample_y = self.transform(sample_x, sample_y)
        return sample_x, sample_y
    