import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, cohen_kappa_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW


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
    def __init__(self, bert_model_name='bert-base-uncased', fine_tune_layers=2):
        super(SiameseBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Freeze all parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        # Unfreeze the last 'fine_tune_layers' encoder layers
        for layer in self.bert.encoder.layer[-fine_tune_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        # We'll take the CLS embeddings and compute their cosine similarity
        # Then we feed that similarity to a small classifier to predict the label.
        self.sim_classifier = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        self.cosine = nn.CosineSimilarity(dim=1)

    def forward(self, q1_input_ids, q1_attention_mask, q2_input_ids, q2_attention_mask):
        # Encode question 1
        q1_outputs = self.bert(
            input_ids=q1_input_ids,
            attention_mask=q1_attention_mask
        )
        q1_cls = q1_outputs.last_hidden_state[:, 0, :]  # [CLS] embedding

        # Encode question 2
        q2_outputs = self.bert(
            input_ids=q2_input_ids,
            attention_mask=q2_attention_mask
        )
        q2_cls = q2_outputs.last_hidden_state[:, 0, :]

        # Compute cosine similarity
        cos_sim = self.cosine(q1_cls, q2_cls).unsqueeze(-1)  # shape: (batch_size, 1)

        # Classifier on similarity
        logits = self.sim_classifier(cos_sim)
        return logits


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
        logits = model(q1_input_ids, q1_attention_mask, q2_input_ids, q2_attention_mask).squeeze(-1)
        loss = criterion(logits, labels)
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

            logits = model(q1_input_ids, q1_attention_mask, q2_input_ids, q2_attention_mask).squeeze(-1)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    # Convert predictions to binary labels
    binary_preds = (np.array(all_preds) >= 0.5).astype(int)
    f1_micro = f1_score(all_labels, binary_preds, average='micro')
    f1_macro = f1_score(all_labels, binary_preds, average='macro')
    cohen = cohen_kappa_score(all_labels, binary_preds)

    return avg_loss, f1_micro, f1_macro, cohen


# ============================================================
# Main Function
# ============================================================
def main():
    # Parameters
    data_path = "../Data/train.csv"  # Update path as needed
    checkpoint_path = "siamese_bert_model_best_better_split.pth"
    bert_model_name = 'bert-base-uncased'
    batch_size = 32
    epochs = 3
    max_len = 128
    fine_tune_layers = 2
    lr = 2e-5

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
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

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

    # Training Loop
    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_loss, f1_micro, f1_macro, cohen = eval_model(model, test_loader, device, criterion)

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

    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
