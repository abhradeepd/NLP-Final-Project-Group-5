# %% Import Libraries
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os
from feature_extraction import process_file_and_extract_features  # Import the provided function

# %% Constants and Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
DIV_ERROR = 0.000001

# %% Data Loading
filename = "Data/train.csv"
size = len(pd.read_csv(filename))
rows_to_train = size

# Use the process_file_and_extract_features to generate the features
processed_data = process_file_and_extract_features(filename, rows_to_train)

# Verify processed data
print(f"Columns in processed data: {processed_data.columns}")

# %% Dataset Class
class QuestionsDataset(Dataset):
    def __init__(self, data):
        self.questions1 = data['question1'].tolist()
        self.questions2 = data['question2'].tolist()

    def __len__(self):
        return len(self.questions1)

    def __getitem__(self, idx):
        return self.questions1[idx], self.questions2[idx]

# %% Function to compute BERT embeddings
def compute_batch_embeddings(batch_sentences):
    inputs = tokenizer(
        batch_sentences,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = bert_model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS embeddings
    return cls_embeddings.cpu().numpy()

# %% Prepare Dataset and DataLoader
dataset = QuestionsDataset(processed_data)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

vecs1, vecs2 = [], []

for batch in tqdm(dataloader, desc="Processing BERT Embeddings"):
    q1_batch, q2_batch = batch

    # Compute BERT embeddings
    vecs1_batch = compute_batch_embeddings(q1_batch)
    vecs2_batch = compute_batch_embeddings(q2_batch)
    vecs1.append(vecs1_batch)
    vecs2.append(vecs2_batch)

# Combine embeddings
vecs1 = np.vstack(vecs1)
vecs2 = np.vstack(vecs2)

# %% Combine Processed Features with BERT Embeddings
all_features = processed_data.drop(columns=['question1', 'question2', 'is_duplicate']).values  # Drop text and label columns

# Include labels (is_duplicate)
labels = processed_data['is_duplicate'].values

# %% Save Combined Data to .pt File
os.makedirs('Data', exist_ok=True)
output_path = 'Data/merged_features_embeddings.pt'

data_to_save = {
    'id': torch.tensor(processed_data['id'].values),  # Assuming 'id' column exists in processed_data
    'q1_feats_bert': torch.tensor(vecs1, dtype=torch.float32),
    'q2_feats_bert': torch.tensor(vecs2, dtype=torch.float32),
    'features': torch.tensor(all_features, dtype=torch.float32),
    'labels': torch.tensor(labels, dtype=torch.float32)  # Add labels as tensor
}

torch.save(data_to_save, output_path)
print(f"Data saved to {output_path}")

# %% Verify Saved Data
loaded_data = torch.load(output_path)
print("Keys:", loaded_data.keys())
print("First ID:", loaded_data['id'][0])
print("First Q1 Embedding Shape:", loaded_data['q1_feats_bert'].shape)
print("First Q2 Embedding Shape:", loaded_data['q2_feats_bert'].shape)
print("First Feature Vector Length:", len(loaded_data['features'][0]))
print("First Label:", loaded_data['labels'][0])
