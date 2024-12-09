import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

from data_loader import load_data, TextPairDataset
from model import BertForTextPairClassification
from train import train_epoch
from evaluate import eval_model
def main():
    # Parameters
    DATA_FILE_TRAIN = '../Data/final_train.csv'  # Replace with your data file path
    DATA_FILE_VAL = '../Data/final_val.csv'
    PRETRAINED_MODEL_NAME = 'bert-base-uncased'
    MAX_LENGTH = 256
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    MODEL_PATH = 'best_model_state_bert_pair_mlp2_gc.pt'
    OPTIMIZER_PATH = 'optimizer_state_gc.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess data
    #df = load_data(DATA_FILE)

    # Split data into train+val and test
    #train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

    # Further split train+val into train and val
    #train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42)  # Adjust size as needed
    train_df = load_data(DATA_FILE_TRAIN)
    val_df = load_data(DATA_FILE_VAL)
    # Use the tokenizer and dataset classes for preprocessing
    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL_NAME)

    train_dataset = TextPairDataset(train_df, tokenizer, max_length=MAX_LENGTH)
    val_dataset = TextPairDataset(val_df, tokenizer, max_length=MAX_LENGTH)

    # Create DataLoader for each set
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    #test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Output sizes for verification
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    # Initialize model
    model = BertForTextPairClassification(pretrained_model_name=PRETRAINED_MODEL_NAME, drop_out=0.21,num_labels=2, freeze_layers=10)
    model = model.to(device)

    # Optimizer and scheduler
    #optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.AdamW([
        {'params': model.bert.encoder.layer[10].intermediate.dense.weight, 'lr': 1e-6},
        {'params': [p for n, p in model.named_parameters() if
                    'bert.encoder.layer.10.intermediate.dense.weight' not in n]}
    ], lr=LEARNING_RATE)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Load model and optimizer if they exist
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        if os.path.exists(OPTIMIZER_PATH):
            print(f"Loading optimizer from {OPTIMIZER_PATH}")
            checkpoint = torch.load(OPTIMIZER_PATH, map_location=device)
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_loss']
        else:
            print("No optimizer state found. Starting from scratch.")
            start_epoch = 0
            best_loss = float('inf')
    else:
        print("No pre-trained model found. Training from scratch.")
        start_epoch = 0
        best_loss = float('inf')
    scaler = torch.cuda.amp.GradScaler()
    # # Training loop
    # for epoch in range(start_epoch, EPOCHS):
    #     print(f'Epoch {epoch + 1}/{EPOCHS}')
    #
    #     train_loss = train_epoch(model, train_loader, optimizer, device, scheduler,scaler)
    #     print(f'Training loss: {train_loss}')
    #
    #     # Evaluation
    #     acc, f1_macro, f1_micro, cohen_kappa, avg_loss, auc_score, conf_matrix = eval_model(model, val_loader, device)
    #     print(f'Validation Loss {avg_loss}')
    #     print(f'Validation Accuracy: {acc}, F1 Macro Score: {f1_macro}, F1 Micro Score: {f1_micro}, Cohen Kappa: {cohen_kappa}')
    #
    #     # Save the best model and optimizer state
    #     if avg_loss < best_loss:
    #         torch.save(model.state_dict(), MODEL_PATH)
    #         torch.save({
    #             'optimizer_state': optimizer.state_dict(),
    #             'scheduler_state': scheduler.state_dict(),
    #             'epoch': epoch,
    #             'best_loss': avg_loss
    #         }, OPTIMIZER_PATH)
    #         best_loss = avg_loss
    #         print(f"Saved the best model and optimizer state to {MODEL_PATH} and {OPTIMIZER_PATH}")
    #
    # print('Training complete!')
    # Training loop
    # Training loop
    for epoch in range(start_epoch, EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')

        train_loss, avg_grad_norm_before, avg_grad_norm_after = train_epoch(
            model, train_loader, optimizer, device, scheduler, scaler
        )
        print(f'Training loss: {train_loss:.4f}')
        print(f'Average Grad Norm Before Clipping: {avg_grad_norm_before:.4f}')
        print(f'Average Grad Norm After Clipping: {avg_grad_norm_after:.4f}')

        # Evaluation
        acc, f1_macro, f1_micro, cohen_kappa, val_loss, auc_score, conf_matrix = eval_model(model, val_loader, device)
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {acc:.4f}, F1 Macro: {f1_macro:.4f}, '
              f'F1 Micro: {f1_micro:.4f}, Cohen Kappa: {cohen_kappa:.4f}, AUC: {auc_score:.4f}')

        # Save the best model and optimizer state
        if val_loss < best_loss:
            torch.save(model.state_dict(), MODEL_PATH)
            torch.save({
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'epoch': epoch,
                'best_loss': val_loss
            }, OPTIMIZER_PATH)
            best_loss = val_loss
            print(f"Saved the best model and optimizer state to {MODEL_PATH} and {OPTIMIZER_PATH}")

    print('Training complete!')


if __name__ == '__main__':
    main()
