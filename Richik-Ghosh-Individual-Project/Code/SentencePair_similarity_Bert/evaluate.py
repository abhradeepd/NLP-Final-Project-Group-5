import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix

def eval_model(model, data_loader, device):
    model = model.eval()
    preds = []
    labels_list = []
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss+=loss.item()
            # Apply sigmoid and threshold logits for binary classification
            # probabilities = torch.sigmoid(logits)
            # predictions = (probabilities > 0.5).long()

            _, predictions = torch.max(logits, dim=1) #for softmax
            preds.extend(predictions.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    acc = accuracy_score(labels_list, preds)
    f1_macro = f1_score(labels_list, preds, average='macro')
    f1_micro = f1_score(labels_list, preds, average='micro')
    cohen_kappa = cohen_kappa_score(labels_list, preds)
    # AUC
    auc_score = roc_auc_score(labels_list, preds)

    # Confusion Matrix
    conf_matrix = confusion_matrix(labels_list, preds)
    return acc, f1_macro, f1_micro, cohen_kappa, avg_loss, auc_score, conf_matrix
