import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(model, data_loader, optimizer, device, scheduler, scaler):
    """
    Train the model for one epoch using mixed precision.

    Args:
        model: The model to train.
        data_loader: DataLoader for training data.
        optimizer: Optimizer for updating model parameters.
        device: Device to run training on (CPU/GPU).
        scheduler: Learning rate scheduler.
        scaler: GradScaler for mixed precision training.

    Returns:
        avg_loss: Average loss over the epoch.
    """
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader, desc='Training'):
        # Move batch data to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with torch.cuda.amp.autocast():
            loss, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Scheduler step
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss