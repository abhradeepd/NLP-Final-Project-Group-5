
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

class TextPairDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question1 = str(self.data.iloc[idx]['question1'])
        question2 = str(self.data.iloc[idx]['question2'])
        label = self.data.iloc[idx]['label']

        encoding = self.tokenizer(
            question1,
            question2,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        inputs = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

        return inputs

def load_data(file_path):
    df = pd.read_csv(file_path)
    #df =df.head(100)
    df = df[['question1', 'question2', 'is_duplicate']]
    df = df.dropna()
    df['label'] = df['is_duplicate'].astype(int)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the data
    return df
