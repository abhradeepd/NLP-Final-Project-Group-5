import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for q1, q2, features, labels in tqdm(dataloader, desc="Training"):
        q1, q2, features, labels = q1.to(device), q2.to(device), features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(q1, q2, features)
        loss = criterion(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_predictions = []
    all_outputs = []

    with torch.no_grad():
        for q1, q2, features, labels in tqdm(dataloader, desc="Evaluating"):
            q1, q2, features, labels = q1.to(device), q2.to(device), features.to(device), labels.to(device)

            outputs = model(q1, q2, features).squeeze()
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()

            predictions = (outputs > 0.5).float()  # Convert logits to binary predictions
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average="macro")
    f1_micro = f1_score(all_labels, all_predictions, average="micro")
    auc_score = roc_auc_score(all_labels, all_outputs)

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "auc": auc_score
    }
