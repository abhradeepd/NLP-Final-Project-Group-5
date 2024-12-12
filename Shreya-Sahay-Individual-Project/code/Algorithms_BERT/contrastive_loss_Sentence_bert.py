import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, cohen_kappa_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW


# ============================================================
# Dataset
# ============================================================
class SiameseQuestionsDataset(Dataset):
    def __init__(self, questions1, questions2, labels, tokenizer, max_len=128):
        """
        Dataset for pairs of questions and binary labels.
        We will tokenize and encode on-the-fly.
        """
        self.questions1 = questions1
        self.questions2 = questions2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        q1 = self.questions1[idx]
        q2 = self.questions2[idx]
        label = self.labels[idx]

        encoded_q1 = self.tokenizer(
            q1,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        encoded_q2 = self.tokenizer(
            q2,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'q1_input_ids': encoded_q1['input_ids'].squeeze(0),
            'q1_attention_mask': encoded_q1['attention_mask'].squeeze(0),
            'q2_input_ids': encoded_q2['input_ids'].squeeze(0),
            'q2_attention_mask': encoded_q2['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }


# ============================================================
# Model
# ============================================================
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        """
        Contrastive Loss using cosine distance.
        L = (1 - Y) * 0.5 * D^2 + Y * 0.5 * max(0, margin - D)^2
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.cosine = nn.CosineSimilarity(dim=1)

    def forward(self, q1_embeddings, q2_embeddings, labels):
        # Normalize embeddings for stable cosine similarity
        q1_embeddings = nn.functional.normalize(q1_embeddings, p=2, dim=1)
        q2_embeddings = nn.functional.normalize(q2_embeddings, p=2, dim=1)

        # Compute cosine similarity
        cos_sim = self.cosine(q1_embeddings, q2_embeddings)
        # Convert similarity to distance
        cos_dist = 1 - cos_sim

        # Compute loss
        positive_loss = labels * (cos_dist ** 2)
        negative_loss = (1 - labels) * torch.clamp(self.margin - cos_dist, min=0) ** 2
        losses = 0.5 * (positive_loss + negative_loss)

        return losses.mean()


class SiameseSBERTModel(nn.Module):
    def __init__(self, sbert_model_name='sentence-transformers/paraphrase-MiniLM-L6-v2'):
        super(SiameseSBERTModel, self).__init__()
        self.sbert = AutoModel.from_pretrained(sbert_model_name)
        self.hidden_dim = self.sbert.config.hidden_size  # Dynamically retrieve hidden size (e.g., 384)

        # MLP to reduce the embedding dimension
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 4, self.hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, q1_input_ids, q1_attention_mask, q2_input_ids, q2_attention_mask):
        # Encode question 1
        q1_outputs = self.sbert(
            input_ids=q1_input_ids,
            attention_mask=q1_attention_mask
        )
        q1_cls = q1_outputs.last_hidden_state[:, 0, :]  # [CLS] embedding

        # Encode question 2
        q2_outputs = self.sbert(
            input_ids=q2_input_ids,
            attention_mask=q2_attention_mask
        )
        q2_cls = q2_outputs.last_hidden_state[:, 0, :]

        # Reduce dimensions using MLP
        q1_reduced = self.mlp(q1_cls)
        q2_reduced = self.mlp(q2_cls)

        return q1_reduced, q2_reduced



# ============================================================
# Training & Evaluation Functions
# ============================================================
def train_one_epoch(model, dataloader, optimizer, device, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        q1_input_ids = batch['q1_input_ids'].to(device)
        q1_attention_mask = batch['q1_attention_mask'].to(device)
        q2_input_ids = batch['q2_input_ids'].to(device)
        q2_attention_mask = batch['q2_attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        q1_embeddings, q2_embeddings = model(q1_input_ids, q1_attention_mask, q2_input_ids, q2_attention_mask)
        loss = criterion(q1_embeddings, q2_embeddings, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def eval_model(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            q1_input_ids = batch['q1_input_ids'].to(device)
            q1_attention_mask = batch['q1_attention_mask'].to(device)
            q2_input_ids = batch['q2_input_ids'].to(device)
            q2_attention_mask = batch['q2_attention_mask'].to(device)
            labels = batch['label'].to(device)

            q1_embeddings, q2_embeddings = model(q1_input_ids, q1_attention_mask, q2_input_ids, q2_attention_mask)
            loss = criterion(q1_embeddings, q2_embeddings, labels)
            total_loss += loss.item()

            cos_sim = nn.CosineSimilarity(dim=1)(q1_embeddings, q2_embeddings).cpu().numpy()
            preds = (cos_sim >= 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    cohen = cohen_kappa_score(all_labels, all_preds)

    return avg_loss, f1_micro, f1_macro, cohen


# ============================================================
# Main Function
# ============================================================
def main():
    # Parameters
    data_path = "../Data/train.csv"
    checkpoint_path = "sbert_siamese_model.pth"
    sbert_model_name = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
    batch_size = 32
    epochs = 8
    max_len = 128
    lr = 1e-5

    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    df['question1'] = df['question1'].astype(str)
    df['question2'] = df['question2'].astype(str)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df[['question1', 'question2']],
        df['is_duplicate'],
        test_size=0.15,
        stratify=df['is_duplicate'],
        random_state=42
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(sbert_model_name)

    # Datasets and Dataloaders
    train_dataset = SiameseQuestionsDataset(
        questions1=X_train['question1'].values,
        questions2=X_train['question2'].values,
        labels=y_train.values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    test_dataset = SiameseQuestionsDataset(
        questions1=X_test['question1'].values,
        questions2=X_test['question2'].values,
        labels=y_test.values,
        tokenizer=tokenizer,
        max_len=max_len
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model, Loss, Optimizer
    model = SiameseSBERTModel(sbert_model_name=sbert_model_name).to(device)
    criterion = ContrastiveLoss(margin=0.8)
    optimizer = AdamW(model.parameters(), lr=lr)

    # Training Loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_loss, f1_micro, f1_macro, cohen = eval_model(model, test_loader, device, criterion)

        print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.4f}")
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")
        print(f"Epoch {epoch + 1}, F1 Micro: {f1_micro:.4f}, F1 Macro: {f1_macro:.4f}, Cohen's Kappa: {cohen:.4f}")

        # Save the best model
        if epoch == 0 or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[INFO] Model saved at epoch {epoch + 1}")

    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
