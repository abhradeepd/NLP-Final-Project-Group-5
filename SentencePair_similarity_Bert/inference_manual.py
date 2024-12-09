import torch
from transformers import BertTokenizerFast
from model import BertForTextPairClassification

def load_model(model_path, pretrained_model_name, device, num_labels):
    model = BertForTextPairClassification(pretrained_model_name=pretrained_model_name, drop_out=0.1, num_labels=num_labels, freeze_layers=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_text(pair, tokenizer, max_length):
    inputs = tokenizer(
        pair[0],
        pair[1],
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    return inputs

def predict(model, tokenizer, text_pair, device, max_length=256):
    inputs = preprocess_text(text_pair, tokenizer, max_length)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)[1]
        _, prediction = torch.max(logits, dim=1)
    return prediction.cpu().item()

if __name__ == '__main__':
    # Parameters
    MODEL_PATH = 'best_model_state_bert_pair_mlp2.pt'
    PRETRAINED_MODEL_NAME = 'bert-base-uncased'
    MAX_LENGTH = 256
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_LABELS = 2  # Adjust based on your dataset

    # Load tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL_NAME)
    model = load_model(MODEL_PATH, PRETRAINED_MODEL_NAME, DEVICE, NUM_LABELS)

    # Input manual text pairs
    while True:
        text1 = input("Enter the first text: ")
        text2 = input("Enter the second text: ")
        if text1.lower() == 'exit' or text2.lower() == 'exit':
            print("Exiting inference...")
            break
        text_pair = (text1, text2)
        prediction = predict(model, tokenizer, text_pair, DEVICE, MAX_LENGTH)
        print(f'Predicted class: {prediction}')
