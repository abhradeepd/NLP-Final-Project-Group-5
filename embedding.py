# %% Import required libraries
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

# %% Importing the data
df = pd.read_csv('Data/train.csv')
print(len(df))
# Use a subset of the data for demonstration
num_rows = len(df)
df = df[:num_rows]
#%%
print(df.head())
#%%
# Ensure all questions are strings
df['question1'] = df['question1'].apply(lambda x: str(x))
df['question2'] = df['question2'].apply(lambda x: str(x))

# %% Initialize BERT tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

# %% Create a custom Dataset class
class QuestionsDataset(Dataset):
    def __init__(self, questions1, questions2):
        self.questions1 = questions1
        self.questions2 = questions2

    def __len__(self):
        return len(self.questions1)

    def __getitem__(self, idx):
        return self.questions1[idx], self.questions2[idx]

# Prepare dataset and DataLoader
dataset = QuestionsDataset(df['question1'].tolist(), df['question2'].tolist())
batch_size = 8  # Adjust based on your GPU memory
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# %% Function to compute BERT embeddings for a batch
def compute_batch_embeddings(batch_sentences):
    # Tokenize and encode the batch of sentences
    inputs = tokenizer(
        batch_sentences,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    # Forward pass through BERT
    with torch.no_grad():
        outputs = bert_model(**inputs)
        # Extract the [CLS] token embeddings
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)

    return cls_embeddings.cpu().numpy()  # Convert to NumPy array

# %% Compute embeddings for question1 and question2 in batches
vecs1 = []
vecs2 = []

for batch in tqdm(dataloader, desc="Processing Batches"):
    qu1_batch, qu2_batch = batch

    # Compute BERT embeddings for question1 and question2
    vecs1_batch = compute_batch_embeddings(qu1_batch)
    vecs2_batch = compute_batch_embeddings(qu2_batch)

    vecs1.append(vecs1_batch)
    vecs2.append(vecs2_batch)

# Combine all batches into single arrays
vecs1 = np.vstack(vecs1)
vecs2 = np.vstack(vecs2)

# Add BERT embeddings to the DataFrame
df['q1_feats_bert'] = list(vecs1)
df['q2_feats_bert'] = list(vecs2)

# %% Output the DataFrame
print(df[['question1', 'question2', 'q1_feats_bert', 'q2_feats_bert']])
#%%
df.to_csv('Data/bert_embeddings.csv', index=False)

