import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
#
#
# def train_epoch(model, data_loader, optimizer, device, scheduler, scaler):
#     """
#     Train the model for one epoch using mixed precision.
#
#     Args:
#         model: The model to train.
#         data_loader: DataLoader for training data.
#         optimizer: Optimizer for updating model parameters.
#         device: Device to run training on (CPU/GPU).
#         scheduler: Learning rate scheduler.
#         scaler: GradScaler for mixed precision training.
#
#     Returns:
#         avg_loss: Average loss over the epoch.
#     """
#     model.train()
#     total_loss = 0
#
#     for batch in tqdm(data_loader, desc='Training'):
#         # Move batch data to device
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#
#         optimizer.zero_grad()
#
#         # Mixed precision forward pass
#         with torch.cuda.amp.autocast():
#             loss, _ = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 labels=labels
#             )
#
#         # Backward pass with gradient scaling
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#
#         # Scheduler step
#         scheduler.step()
#
#         total_loss += loss.item()
#
#     avg_loss = total_loss / len(data_loader)
#     return avg_loss
def train_epoch(model, data_loader, optimizer, device, scheduler, scaler, max_grad_norm=4.0):
    model.train()
    total_loss = 0
    total_grad_norm_before = 0
    total_grad_norm_after = 0
    batch_count = 0

    for batch_idx, batch in enumerate(tqdm(data_loader, desc='Training')):
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

        # Check loss for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[Batch {batch_idx + 1}] Problematic Loss Detected: {loss.item()}")
            scaler.update()  # Advance the scaler state even if skipping
            continue

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Now unscale the gradients once
        scaler.unscale_(optimizer)

        # Check gradients after unscaling
        invalid_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"[Batch {batch_idx + 1}] Gradient issue in parameter: {name}")
                    invalid_grad = True
                    break

        if invalid_grad:
            print(f"[Batch {batch_idx + 1}] Skipping batch due to invalid gradients.")
            optimizer.zero_grad()  # Clear invalid gradients
            scaler.update()  # Advance the scaler state even if skipping
            continue

        # If gradients are valid, proceed with gradient clipping
        total_norm_before = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_before += param_norm.item() ** 2
        total_grad_norm_before += (total_norm_before ** 0.5)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        total_norm_after = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_after += param_norm.item() ** 2
        total_grad_norm_after += (total_norm_after ** 0.5)

        batch_count += 1

        # Optimizer step and scaler update
        scaler.step(optimizer)
        scaler.update()

        # Scheduler step
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    avg_grad_norm_before = total_grad_norm_before / batch_count if batch_count > 0 else 0
    avg_grad_norm_after = total_grad_norm_after / batch_count if batch_count > 0 else 0

    return avg_loss, avg_grad_norm_before, avg_grad_norm_after
