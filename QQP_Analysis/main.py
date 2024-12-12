import torch
import os
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.nn import BCELoss, BCEWithLogitsLoss
from train_eval import train_one_epoch, evaluate
from model import BaseModel
from dataloader import load_data

# File paths
DATA_PATH = "../Data/merged_features_embeddings.pt"
OUTPUT_DIR = "Data"
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model_lstm.pth")

# Hyperparameters
BATCH_SIZE = 32
FEATURE_DIM = 768
HIDDEN_DIM = 256
HEAD_TYPE = "lstm"  # Can be "lstm", "gru", or "cnn"
LEARNING_RATE = 0.00001
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the original .pt file
data = torch.load(DATA_PATH)

# Extract data components
ids = data['id']
q1_feats = data['q1_feats_bert']
q2_feats = data['q2_feats_bert']
features = data['features']
labels = data['labels']

# Split data into train, validation, and test
train_idx, temp_idx = train_test_split(range(len(ids)), test_size=0.15, random_state=42)  # 15% for val + test
val_idx, test_idx = train_test_split(temp_idx, test_size=2/3, random_state=42)  # 2/3 of 15% for test = 10%

def split_data(indices):
    return {
        'id': ids[indices],
        'q1_feats_bert': q1_feats[indices],
        'q2_feats_bert': q2_feats[indices],
        'features': features[indices],
        'labels': labels[indices]
    }

# Create splits
train_data = split_data(train_idx)
val_data = split_data(val_idx)
test_data = split_data(test_idx)

# Save the test data separately for inference
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.save(test_data, os.path.join(OUTPUT_DIR, "test_split.pt"))



# Load train and validation data
train_loader = load_data(train_data, batch_size=BATCH_SIZE)
val_loader = load_data(val_data, batch_size=BATCH_SIZE)

# Initialize model, optimizer, and loss function
model = BaseModel(head_type=HEAD_TYPE, feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
criterion = BCEWithLogitsLoss()

# Training loop with checkpointing
best_val_loss = float("inf")
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

    # Train for one epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    print(f"Training Loss: {train_loss:.4f}")

    # Evaluate on validation data
    metrics = evaluate(model, val_loader, criterion, DEVICE)
    val_loss = metrics["loss"]
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Metrics: Accuracy: {metrics['accuracy']:.4f}, F1-Macro: {metrics['f1_macro']:.4f}, AUC: {metrics['auc']:.4f}")

    # Save best model checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Best model saved with Validation Loss: {best_val_loss:.4f}")

print("Training complete. Best model saved at:", BEST_MODEL_PATH)
