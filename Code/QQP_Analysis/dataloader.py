from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data):
        self.ids = data['id']
        self.q1_feats = data['q1_feats_bert']
        self.q2_feats = data['q2_feats_bert']
        self.features = data['features']
        self.labels = data['labels']

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return (
            self.q1_feats[idx],
            self.q2_feats[idx],
            self.features[idx],
            self.labels[idx]
        )

def load_data(data, batch_size=32):
    dataset = CustomDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
