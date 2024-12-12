import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, cohen_kappa_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer

# ============================================================
# Dataset
# ============================================================
class SiameseQuestionsDataset(Dataset):
    def __init__(self, questions1, questions2, labels, tokenizer, max_len=128):
        """
        Dataset for pairs of questions and binary labels.
        Tokenizes and encodes questions on-the-fly.
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

        encoded_q1 = self.tokenizer.encode_plus(
            q1,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        encoded_q2 = self.tokenizer.encode_plus(
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
class SiameseBertModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', fine_tune_layers=3):
        super(SiameseBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Freeze all layers initially
        for param in self.bert.parameters():
            param.requires_grad = False

        # Unfreeze the last `fine_tune_layers` encoder layers
        for layer in self.bert.encoder.layer[-fine_tune_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Fully connected layer for dimensionality reduction
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )

    def forward(self, q1_input_ids, q1_attention_mask, q2_input_ids, q2_attention_mask):
        # Encode question 1
        q1_outputs = self.bert(
            input_ids=q1_input_ids,
            attention_mask=q1_attention_mask
        )
        q1_cls = q1_outputs.last_hidden_state[:, 0, :]  # CLS token embedding

        # Encode question 2
        q2_outputs = self.bert(
            input_ids=q2_input_ids,
            attention_mask=q2_attention_mask
        )
        q2_cls = q2_outputs.last_hidden_state[:, 0, :]

        # Reduce dimensions using the fully connected layer
        q1_embeddings = self.fc(q1_cls)
        q2_embeddings = self.fc(q2_cls)

        return q1_embeddings, q2_embeddings

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        """
        Contrastive Loss using cosine distance.
        L = (1 - Y) * 0.5 * D^2 + Y * 0.5 * max(0, margin - D)^2
        where Y is the label (1: similar, 0: dissimilar), and D is the cosine distance.
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

        # Return batch mean loss
        return losses.mean()

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

            # Compute cosine similarity
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
    data_path = "../Data/train.csv"  # Update path as needed
    checkpoint_path = "bert_siamese_model_bert_constrastive.pth"
    bert_model_name = 'bert-base-uncased'
    batch_size = 32
    epochs = 8
    lr = 1e-5
    max_len = 128
    fine_tune_layers = 2

    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    df['question1'] = df['question1'].astype(str)
    df['question2'] = df['question2'].astype(str)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df[['question1', 'question2']],
        df['is_duplicate'],
        test_size=0.10,
        stratify=df['is_duplicate'],
        random_state=42
    )
    test_data = X_test.copy()
    test_data['is_duplicate'] = y_test.values

    # Save the test data to a CSV
    test_data_path = "../Data/test_data.csv"
    test_data.to_csv(test_data_path, index=False)

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

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
    model = SiameseBertModel(bert_model_name=bert_model_name, fine_tune_layers=fine_tune_layers).to(device)
    criterion = ContrastiveLoss(margin=0.8)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Load checkpoint if exists
    start_epoch = 0
    best_val_loss = float('inf')
    if os.path.exists(checkpoint_path):
        print(f"[INFO] Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"[INFO] Resuming training from epoch {start_epoch + 1} with best validation loss {best_val_loss:.4f}")

    # Training Metrics Storage
    train_losses = []
    val_losses = []
    f1_macros = []
    val_f1_macros = []

    # Training Loop
    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_loss, f1_micro, f1_macro, cohen = eval_model(model, test_loader, device, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        f1_macros.append(f1_macro)
        val_f1_macros.append(f1_macro)

        print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.4f}")
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")
        print(f"Epoch {epoch + 1}, F1 Micro: {f1_micro:.4f}, F1 Macro: {f1_macro:.4f}, Cohen's Kappa: {cohen:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            print(f"[INFO] Model saved with best validation loss: {best_val_loss:.4f}")

    # Plot Metrics
    plt.figure(figsize=(12, 6))

    # Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss', marker='o')
    plt.title('Train and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # F1 Macro Plot
    plt.subplot(1, 2, 2)
    plt.plot(f1_macros, label='Train F1 Macro')
    plt.plot(val_f1_macros, label='Validation F1 Macro')
    plt.title('F1 Macro vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Macro')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()