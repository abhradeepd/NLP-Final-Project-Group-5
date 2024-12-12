import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel
from sklearn.manifold import TSNE
from tqdm import tqdm

import plotly.graph_objs as go

############################################################
# Data loading and Dataset
############################################################

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
    df = df[['question1', 'question2', 'is_duplicate']]
    df = df.dropna()
    df['label'] = df['is_duplicate'].astype(int)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the data
    return df

############################################################
# Model definition
############################################################

class BertForTextPairClassification(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', num_labels=2, drop_out=0.5, freeze_layers=10):
        super(BertForTextPairClassification, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)

        bert_hidden_size = self.bert.config.hidden_size
        hidden_size1 = bert_hidden_size // 2
        hidden_size2 = bert_hidden_size // 4

        self.drop = nn.Dropout(p=drop_out)
        self.fc1 = nn.Linear(bert_hidden_size, hidden_size1)
        self.drop1 = nn.Dropout(p=drop_out)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.drop2 = nn.Dropout(p=drop_out)
        self.out = nn.Linear(hidden_size2, num_labels)

        # Freeze layers
        for idx, layer in enumerate(self.bert.encoder.layer):
            if idx < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        hidden_output1 = torch.relu(self.fc1(self.drop(pooled_output)))
        hidden_output1 = self.drop1(hidden_output1)
        hidden_output2 = torch.relu(self.fc2(hidden_output1))
        hidden_output2 = self.drop2(hidden_output2)
        logits = self.out(hidden_output2)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return loss, logits

############################################################
# Function to get embeddings from the model (with tqdm)
############################################################

def get_embeddings(model, data_loader, device):
    model.eval()
    embeddings = []
    labels = []
    # Using tqdm to show progress in terminal
    for batch in tqdm(data_loader, desc="Extracting Embeddings"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        batch_labels = batch['labels'].cpu().numpy()

        with torch.no_grad():
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output.cpu().numpy()  # shape: [batch_size, hidden_size]

        embeddings.append(pooled_output)
        labels.extend(batch_labels)

    # Flatten list of batches
    import numpy as np
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    return embeddings, labels
############################################################
# 2D TSNE Visualization for BERT embeddings
############################################################



def plot_embeddings_tsne_3d(embeddings, labels, questions1, questions2, n_iter=1000, perplexity=30):
    """
    Generates a 3D t-SNE visualization of embeddings with labeled legend for binary labels.

    Args:
        embeddings: Numpy array of shape [num_samples, embedding_dim].
        labels: Array of shape [num_samples], 1 for duplicate, 0 for non-duplicate.
        questions1: List of the first question in each pair.
        questions2: List of the second question in each pair.
        n_iter: Number of iterations for t-SNE optimization.

    Returns:
        A Plotly 3D scatter plot.
    """
    print("Running t-SNE on embeddings...")
    tsne = TSNE(n_components=3, random_state=42, verbose=1, n_iter=n_iter, perplexity=perplexity)
    tsne_results = tsne.fit_transform(embeddings)

    x = tsne_results[:, 0]
    y = tsne_results[:, 1]
    z = tsne_results[:, 2]

    # Create hover text combining both questions
    hover_text = [f"Q1: {q1}<br>Q2: {q2}" for q1, q2 in zip(questions1, questions2)]

    # Separate data for duplicate and non-duplicate labels
    duplicate_indices = labels == 1
    non_duplicate_indices = labels == 0

    # Create separate traces for duplicate and non-duplicate
    trace_duplicate = go.Scatter3d(
        x=x[duplicate_indices],
        y=y[duplicate_indices],
        z=z[duplicate_indices],
        mode='markers',
        marker=dict(
            size=6,
            color='red',
            opacity=0.7
        ),
        name='Duplicate',
        text=[hover_text[i] for i in range(len(hover_text)) if duplicate_indices[i]],
        hoverinfo='text'
    )

    trace_non_duplicate = go.Scatter3d(
        x=x[non_duplicate_indices],
        y=y[non_duplicate_indices],
        z=z[non_duplicate_indices],
        mode='markers',
        marker=dict(
            size=6,
            color='blue',
            opacity=0.7
        ),
        name='Non-Duplicate',
        text=[hover_text[i] for i in range(len(hover_text)) if non_duplicate_indices[i]],
        hoverinfo='text'
    )

    # Combine both traces
    fig = go.Figure(data=[trace_duplicate, trace_non_duplicate])
    fig.update_layout(
        height=800,
        width=800,
        title='3D t-SNE Visualization of Question Embeddings',
        scene=dict(
            xaxis=dict(title='Dimension 1'),
            yaxis=dict(title='Dimension 2'),
            zaxis=dict(title='Dimension 3')
        )
    )

    return fig
