# ml_algorithms/tsne_for_question_pairs.py

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import plotly.graph_objs as go

# ============================================================
# Siamese Model Definition (Same as in your training code)
# ============================================================

class SiameseBertModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', fine_tune_layers=3, hidden_dim=768):
        super(SiameseBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Freeze all parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        # Unfreeze the last 'fine_tune_layers' encoder layers
        for layer in self.bert.encoder.layer[-fine_tune_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        # MLPs to reduce the hidden dimension
        self.mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, input_ids, attention_mask):
        # Encode the question
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] embedding

        # Pass through MLPs
        reduced_emb = self.mlp2(self.mlp1(cls_emb))

        return reduced_emb

# ============================================================
# Dataset Definition for Unique Questions
# ============================================================

class UniqueQuestionsDataset(Dataset):
    def __init__(self, questions, labels, tokenizer, max_len=128):
        """
        Dataset for unique questions and their labels.
        """
        self.questions = questions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        label = self.labels[idx]

        encoded = self.tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float),
            'question_text': question
        }

# ============================================================
# Function to Extract Embeddings
# ============================================================

def extract_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []
    questions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['label'].cpu().numpy()
            batch_questions = batch['question_text']

            # Get embeddings
            emb = model(input_ids, attention_mask)
            emb = emb.cpu().numpy()

            embeddings.append(emb)
            labels.extend(batch_labels)
            questions.extend(batch_questions)

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    questions = np.array(questions)

    return embeddings, labels, questions

# ============================================================
# Function to Load Unique Questions
# ==================
# ==========================================

def load_unique_questions(data_file):
    """
    Extract unique questions and assign labels based on majority vote.

    Args:
        data_file: Path to the CSV file containing 'question1', 'question2', 'is_duplicate'.

    Returns:
        unique_questions: List of unique questions.
        unique_labels: List of labels corresponding to unique questions.
    """
    print("Loading and processing data for unique questions...")
    df = pd.read_csv(data_file)
    df['question1'] = df['question1'].astype(str)
    df['question2'] = df['question2'].astype(str)
    df = df[['question1', 'question2', 'is_duplicate']].dropna().reset_index(drop=True)

    # Combine question1 and question2 into a single DataFrame
    all_questions = pd.concat([
        df[['question1', 'is_duplicate']].rename(columns={'question1': 'question'}),
        df[['question2', 'is_duplicate']].rename(columns={'question2': 'question'})
    ], ignore_index=True)

    # Group by question and assign label based on majority vote
    question_label_map = all_questions.groupby('question')['is_duplicate'].agg(lambda x: x.mode()[0] if not x.mode().empty else 0).to_dict()

    unique_questions = list(question_label_map.keys())
    unique_labels = list(question_label_map.values())

    print(f"Total unique questions: {len(unique_questions)}")
    return unique_questions, unique_labels

# ============================================================
# 3D t-SNE Visualization Function for Unique Questions
# ============================================================


from sklearn.cluster import KMeans


def plot_unique_questions_tsne_3d_with_clusters(embeddings, questions, n_iter=1000, perplexity=30, n_clusters=5):
    """
    Generates a 3D t-SNE visualization of unique question embeddings with clustering.

    Args:
        embeddings: Numpy array of shape [num_unique_questions, embedding_dim].
        questions: Array of shape [num_unique_questions], containing question texts.
        n_iter: Number of iterations for t-SNE optimization.
        perplexity: Perplexity parameter for t-SNE.
        n_clusters: Number of clusters for KMeans.

    Returns:
        A Plotly 3D scatter plot figure.
    """
    print("Running t-SNE on unique question embeddings...")
    # Optional: Apply PCA before t-SNE for performance
    pca = PCA(n_components=50, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)

    tsne = TSNE(n_components=3, random_state=42, verbose=1, n_iter=n_iter, perplexity=perplexity)
    tsne_results = tsne.fit_transform(embeddings_pca)

    x = tsne_results[:, 0]
    y = tsne_results[:, 1]
    z = tsne_results[:, 2]

    # Apply KMeans clustering
    print("Clustering embeddings...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Map cluster IDs to colors and names
    unique_clusters = np.unique(cluster_labels)
    color_palette = ['blue', 'red', 'purple', 'yellow', 'cyan', 'magenta', 'orange', 'lime', 'pink', 'teal']
    cluster_colors = [color_palette[i % len(color_palette)] for i in cluster_labels]
    cluster_names = [f"Cluster {i+1}" for i in unique_clusters]

    # Create hover text with the question
    hover_text = questions.tolist()

    # Create a Plotly 3D scatter plot
    traces = []
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        trace = go.Scatter3d(
            x=x[cluster_mask],
            y=y[cluster_mask],
            z=z[cluster_mask],
            mode='markers',
            marker=dict(
                size=5,
                color=color_palette[cluster_id % len(color_palette)],
                opacity=0.8
            ),
            name=cluster_names[cluster_id],
            text=[hover_text[i] for i in range(len(hover_text)) if cluster_labels[i] == cluster_id],
            hoverinfo='text'
        )
        traces.append(trace)

    fig = go.Figure(data=traces)
    fig.update_layout(
        height=800,
        width=800,
        title='3D t-SNE Visualization of Unique Question Embeddings (Clustered)',
        scene=dict(
            xaxis=dict(title='Dimension 1'),
            yaxis=dict(title='Dimension 2'),
            zaxis=dict(title='Dimension 3')
        ),
        legend=dict(
            itemsizing='constant'
        )
    )

    return fig
