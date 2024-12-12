import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
import matplotlib.pyplot as plt

# ============================================================
# Dataset with Dynamic Masking
# ============================================================
class SiameseQuestionsDataset(Dataset):
    def __init__(self, questions1, questions2, labels, tokenizer, max_len=128):
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

        # Extract question words (simplified for demonstration)
        qw0 = self._extract_question_word(q1)
        qw1 = self._extract_question_word(q2)

        return {
            'q1_input_ids': encoded_q1['input_ids'].squeeze(0),
            'q1_attention_mask': encoded_q1['attention_mask'].squeeze(0),
            'q2_input_ids': encoded_q2['input_ids'].squeeze(0),
            'q2_attention_mask': encoded_q2['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float),
            'qw0': torch.tensor(qw0, dtype=torch.float),
            'qw1': torch.tensor(qw1, dtype=torch.float)
        }

    def _extract_question_word(self, text):
        question_words = ["what", "who", "where", "when", "why", "how"]
        tokens = text.lower().split()
        for token in tokens:
            if token in question_words:
                return question_words.index(token) / len(question_words)  # Normalize for simplicity
        return -1  # No question word found

# ============================================================
# Model
# ============================================================
class SBERTSiameseWithSkip(nn.Module):
    def __init__(self, sbert_model_name='sentence-transformers/paraphrase-MiniLM-L6-v2', pooling_strategy='mean', fine_tune_layers=2):
        super(SBERTSiameseWithSkip, self).__init__()
        self.sbert = AutoModel.from_pretrained(sbert_model_name)
        self.pooling_strategy = pooling_strategy.lower()
        self.hidden_dim = self.sbert.config.hidden_size

        # Freeze all layers initially
        for param in self.sbert.parameters():
            param.requires_grad = False

        # Unfreeze the last `fine_tune_layers` layers
        if fine_tune_layers > 0:
            for layer in self.sbert.encoder.layer[-fine_tune_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # Dense layer for question word skip connection
        self.qw_dense = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU()
        )

        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),  # Additional input for question word difference
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 1)  # Output a single logit
        )

    def pool(self, token_embeddings, attention_mask):
        if self.pooling_strategy == 'mean':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        elif self.pooling_strategy == 'max':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to a large negative value
            return torch.max(token_embeddings, dim=1).values
        elif self.pooling_strategy == 'cls':
            return token_embeddings[:, 0]  # Take the [CLS] token
        else:
            raise ValueError("Invalid pooling strategy. Choose from 'mean', 'max', or 'cls'.")

    def encode(self, input_ids, attention_mask):
        outputs = self.sbert(input_ids=input_ids, attention_mask=attention_mask)
        return self.pool(outputs.last_hidden_state, attention_mask)

    def forward(self, q1_input_ids, q1_attention_mask, q2_input_ids, q2_attention_mask, qw_diff):
        q1_embeddings = self.encode(q1_input_ids, q1_attention_mask)
        q2_embeddings = self.encode(q2_input_ids, q2_attention_mask)

        # Question word skip connection
        qw_embedding = self.qw_dense(qw_diff.unsqueeze(1))

        # Concatenate u, v, |u-v|, and question word difference
        combined = torch.cat([q1_embeddings, q2_embeddings, torch.abs(q1_embeddings - q2_embeddings), qw_embedding], dim=1)

        # Pass through classifier
        logits = self.classifier(combined).squeeze(1)
        return logits

# ============================================================
# Training & Evaluation Functions with Metrics Tracking
# ============================================================
def train_one_epoch_with_skip(model, dataloader, optimizer, device, criterion, sigmoid, alpha=0.1, threshold=0.5):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training"):
        q1_input_ids = batch['q1_input_ids'].to(device)
        q1_attention_mask = batch['q1_attention_mask'].to(device)
        q2_input_ids = batch['q2_input_ids'].to(device)
        q2_attention_mask = batch['q2_attention_mask'].to(device)
        qw0 = batch['qw0'].to(device)
        qw1 = batch['qw1'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Compute question word difference
        qw_diff = torch.abs(qw0 - qw1)

        # Forward pass
        logits = model(q1_input_ids, q1_attention_mask, q2_input_ids, q2_attention_mask, qw_diff)

        # Compute loss
        loss = criterion(logits, labels)

        # Apply question word penalty
        penalty = alpha * torch.mean(torch.abs(qw_diff) * logits)
        loss += penalty

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # For Metrics
        probs = sigmoid(logits)
        preds = (probs >= threshold).long().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, f1, accuracy

def eval_model_with_skip(model, dataloader, device, criterion, sigmoid, alpha=0.1, threshold=0.5):
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
            qw0 = batch['qw0'].to(device)
            qw1 = batch['qw1'].to(device)
            labels = batch['label'].to(device)

            # Compute question word difference
            qw_diff = torch.abs(qw0 - qw1)

            # Forward pass
            logits = model(q1_input_ids, q1_attention_mask, q2_input_ids, q2_attention_mask, qw_diff)

            # Compute loss
            loss = criterion(logits, labels)

            # Apply question word penalty
            penalty = alpha * torch.mean(torch.abs(qw_diff) * logits)
            loss += penalty

            total_loss += loss.item()

            # For Metrics
            probs = sigmoid(logits)
            preds = (probs >= threshold).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, f1, accuracy

# ============================================================
# Main Function with Visualization
# ============================================================
def main():
    # Parameters
    data_path = "../Data/train.csv"
    test_save_path = "../Data/test_data.csv"
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    sbert_model_name = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
    batch_size = 32
    epochs = 5
    max_len = 128
    lr = 3e-5
    alpha = 0.1

    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    df['question1'] = df['question1'].astype(str)
    df['question2'] = df['question2'].astype(str)

    # Split data: 90% Train, 5% Validation, 5% Test
    train_df, temp_df = train_test_split(df, test_size=0.10, stratify=df['is_duplicate'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['is_duplicate'], random_state=42)

    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")

    # Save the test data
    test_df.to_csv(test_save_path, index=False)
    print(f"Test data saved to {test_save_path}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(sbert_model_name)

    # Datasets and Dataloaders
    train_dataset = SiameseQuestionsDataset(
        questions1=train_df['question1'].values,
        questions2=train_df['question2'].values,
        labels=train_df['is_duplicate'].values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    val_dataset = SiameseQuestionsDataset(
        questions1=val_df['question1'].values,
        questions2=val_df['question2'].values,
        labels=val_df['is_duplicate'].values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sigmoid = nn.Sigmoid()

    # Initialize the model
    model = SBERTSiameseWithSkip(
        sbert_model_name=sbert_model_name,
        pooling_strategy='mean',
        fine_tune_layers=2
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr)

    results = {
        "epoch": [],
        "train_loss": [],
        "train_f1": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_f1": [],
        "val_accuracy": []
    }

    best_val_loss = float('inf')
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_model_skip.pt")

    # Load checkpoint if exists
    if os.path.exists(best_checkpoint_path):
        print(f"Loading checkpoint from {best_checkpoint_path}")
        model.load_state_dict(torch.load(best_checkpoint_path))

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Training
        train_loss, train_f1, train_acc = train_one_epoch_with_skip(
            model, train_loader, optimizer, device, criterion, sigmoid, alpha
        )

        # Validation
        val_loss, val_f1, val_acc = eval_model_with_skip(
            model, val_loader, device, criterion, sigmoid, alpha
        )

        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}")

        # Store results
        results["epoch"].append(epoch + 1)
        results["train_loss"].append(train_loss)
        results["train_f1"].append(train_f1)
        results["train_accuracy"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_f1"].append(val_f1)
        results["val_accuracy"].append(val_acc)

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"Saved best model checkpoint at epoch {epoch + 1}")

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(results["epoch"], results["train_loss"], label="Train Loss")
    plt.plot(results["epoch"], results["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(results["epoch"], results["train_f1"], label="Train F1")
    plt.plot(results["epoch"], results["val_f1"], label="Validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Training and Validation F1 Score")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(results["epoch"], results["train_accuracy"], label="Train Accuracy")
    plt.plot(results["epoch"], results["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()