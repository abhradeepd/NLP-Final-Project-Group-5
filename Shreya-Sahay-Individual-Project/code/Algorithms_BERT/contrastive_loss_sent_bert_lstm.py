import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score

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


class SiameseLSTMGRUModel(nn.Module):
    def __init__(self, sbert_model_name='sentence-transformers/paraphrase-MiniLM-L6-v2',
                 rnn_type='LSTM', hidden_size=128, num_layers=1, dropout=0.1):
        super(SiameseLSTMGRUModel, self).__init__()
        self.sbert = AutoModel.from_pretrained(sbert_model_name)
        self.hidden_dim = self.sbert.config.hidden_size  # e.g., 384

        # Choose RNN type
        if rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.hidden_dim, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        elif rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(input_size=self.hidden_dim, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        else:
            raise ValueError("rnn_type must be either 'LSTM' or 'GRU'")

        # MLP to reduce the embedding dimension
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Classification layer
        # Combine both question representations
        self.classifier = nn.Sequential(
            nn.Linear((hidden_size // 2) * 4, (hidden_size // 2) * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear((hidden_size // 2) * 2, 1)  # Output a single logit
        )

    def encode(self, input_ids, attention_mask):
        # Get token embeddings from SBERT
        outputs = self.sbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # outputs.last_hidden_state: [batch_size, seq_len, hidden_size]
        return outputs.last_hidden_state

    def forward(self, q1_input_ids, q1_attention_mask, q2_input_ids, q2_attention_mask):
        # Encode both questions
        q1_embeddings = self.encode(q1_input_ids, q1_attention_mask)  # [batch_size, seq_len, hidden_size]
        q2_embeddings = self.encode(q2_input_ids, q2_attention_mask)

        # Pass through RNN
        q1_output, q1_hidden = self.rnn(q1_embeddings)  # q1_hidden: [num_layers * num_directions, batch, hidden_size]
        q2_output, q2_hidden = self.rnn(q2_embeddings)

        # Concatenate the final forward and backward hidden states for each question
        if isinstance(q1_hidden, tuple):  # LSTM returns (hidden, cell)
            q1_hidden = q1_hidden[0]
            q2_hidden = q2_hidden[0]

        # Assuming num_layers=1 and bidirectional=True
        q1_final = torch.cat((q1_hidden[-2], q1_hidden[-1]), dim=1)  # [batch, hidden_size * 2]
        q2_final = torch.cat((q2_hidden[-2], q2_hidden[-1]), dim=1)  # [batch, hidden_size * 2]

        # Reduce dimensions using MLP
        q1_reduced = self.mlp(q1_final)  # [batch, hidden_size//2]
        q2_reduced = self.mlp(q2_final)  # [batch, hidden_size//2]

        # Combine both question representations
        combined = torch.cat([
            q1_reduced,
            q2_reduced,
            torch.abs(q1_reduced - q2_reduced),
            q1_reduced * q2_reduced
        ], dim=1)  # [batch, hidden_size * 2]

        # Pass through classifier
        logits = self.classifier(combined)  # [batch, 1]
        logits = logits.squeeze(1)  # [batch]

        return q1_reduced, q2_reduced, logits  # Return embeddings and logits


# ============================================================
# Training & Evaluation Functions
# ============================================================
def train_one_epoch(model, dataloader, optimizer, scheduler, device,
                    contrastive_criterion, classification_criterion,
                    contrastive_weight=1.0, classification_weight=1.0,
                    max_grad_norm=1.0):
    model.train()
    total_contrastive_loss = 0
    total_classification_loss = 0

    # Phase 1: Alignment
    print("Training Phase 1: Alignment")
    for batch in tqdm(dataloader, desc="Alignment Training"):
        q1_input_ids = batch['q1_input_ids'].to(device)
        q1_attention_mask = batch['q1_attention_mask'].to(device)
        q2_input_ids = batch['q2_input_ids'].to(device)
        q2_attention_mask = batch['q2_attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        q1_embeddings, q2_embeddings, _ = model(q1_input_ids, q1_attention_mask, q2_input_ids, q2_attention_mask)

        # Compute Contrastive Loss
        contrastive_loss = contrastive_criterion(q1_embeddings, q2_embeddings, labels)
        contrastive_loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()

        total_contrastive_loss += contrastive_loss.item()

    avg_contrastive_loss = total_contrastive_loss / len(dataloader)
    print(f"Phase 1 - Average Contrastive Loss: {avg_contrastive_loss:.4f}")

    # Phase 2: Classification
    print("Training Phase 2: Classification")
    for batch in tqdm(dataloader, desc="Classification Training"):
        q1_input_ids = batch['q1_input_ids'].to(device)
        q1_attention_mask = batch['q1_attention_mask'].to(device)
        q2_input_ids = batch['q2_input_ids'].to(device)
        q2_attention_mask = batch['q2_attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        _, _, logits = model(q1_input_ids, q1_attention_mask, q2_input_ids, q2_attention_mask)

        # Compute Classification Loss
        classification_loss = classification_criterion(logits, labels)
        classification_loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()

        total_classification_loss += classification_loss.item()

    avg_classification_loss = total_classification_loss / len(dataloader)
    print(f"Phase 2 - Average Classification Loss: {avg_classification_loss:.4f}")

    return avg_contrastive_loss, avg_classification_loss


def eval_model(model, dataloader, device, contrastive_criterion, classification_criterion,
               threshold=0.5, contrastive_weight=1.0, classification_weight=1.0):
    model.eval()
    total_contrastive_loss = 0
    total_classification_loss = 0
    all_preds = []
    all_labels = []

    sigmoid = nn.Sigmoid()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            q1_input_ids = batch['q1_input_ids'].to(device)
            q1_attention_mask = batch['q1_attention_mask'].to(device)
            q2_input_ids = batch['q2_input_ids'].to(device)
            q2_attention_mask = batch['q2_attention_mask'].to(device)
            labels = batch['label'].to(device)

            q1_embeddings, q2_embeddings, logits = model(q1_input_ids, q1_attention_mask, q2_input_ids,
                                                         q2_attention_mask)

            # Compute losses
            contrastive_loss = contrastive_criterion(q1_embeddings, q2_embeddings, labels)
            classification_loss = classification_criterion(logits, labels)
            loss = contrastive_weight * contrastive_loss + classification_weight * classification_loss

            total_contrastive_loss += contrastive_loss.item()
            total_classification_loss += classification_loss.item()

            probs = sigmoid(logits)
            preds = (probs >= threshold).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_contrastive_loss = total_contrastive_loss / len(dataloader)
    avg_classification_loss = total_classification_loss / len(dataloader)
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    cohen = cohen_kappa_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_contrastive_loss, avg_classification_loss, f1_micro, f1_macro, cohen, accuracy


# ============================================================
# Main Function
# ============================================================
def main():
    # Parameters
    data_path = "../Data/train.csv"
    checkpoint_path = "siamese_lstmgru_contrastive_model.pth"
    sbert_model_name = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
    batch_size = 32
    epochs = 8
    max_len = 128
    lr = 1e-5
    rnn_type = 'LSTM'  # Choose between 'LSTM' or 'GRU'
    hidden_size = 128
    num_layers = 1
    dropout = 0.1
    threshold = 0.5
    contrastive_weight = 1.0
    classification_weight = 1.0

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
    print(f"Using device: {device}")

    # Model, Loss, Optimizer
    model = SiameseLSTMGRUModel(
        sbert_model_name=sbert_model_name,
        rnn_type=rnn_type,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    contrastive_criterion = ContrastiveLoss(margin=0.8)
    classification_criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr)

    # Scheduler
    total_steps = len(train_loader) * epochs * 2  # Two phases per epoch
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_steps // 2, gamma=0.1)

    best_f1_macro = 0.0  # Initialize best metric

    # Training Loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train for this epoch
        train_contrastive_loss, train_classification_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            contrastive_criterion, classification_criterion,
            contrastive_weight, classification_weight
        )
        print(
            f"Training - Contrastive Loss: {train_contrastive_loss:.4f}, Classification Loss: {train_classification_loss:.4f}")

        # Evaluate on validation set
        val_contrastive_loss, val_classification_loss, f1_micro, f1_macro, cohen, accuracy = eval_model(
            model, test_loader, device, contrastive_criterion, classification_criterion,
            threshold, contrastive_weight, classification_weight
        )

        print(
            f"Validation - Contrastive Loss: {val_contrastive_loss:.4f}, Classification Loss: {val_classification_loss:.4f}")
        print(
            f"F1 Micro: {f1_micro:.4f}, F1 Macro: {f1_macro:.4f}, Cohen's Kappa: {cohen:.4f}, Accuracy: {accuracy:.4f}")

        # Save the best model based on F1 Macro score
        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[INFO] Best model saved at epoch {epoch + 1} with F1 Macro: {f1_macro:.4f}")

    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
