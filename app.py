import streamlit as st
import torch
from transformers import BertTokenizerFast
from model import BertForTextPairClassification

# Predefined constants
MODEL_PATH = "best_model_state_bert_pair_mlp2.pt"
PRETRAINED_MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 256
NUM_LABELS = 2  # Adjust based on your dataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the model
@st.cache_resource
def load_model(model_path, pretrained_model_name, device, num_labels):
    model = BertForTextPairClassification(pretrained_model_name=pretrained_model_name, drop_out=0.1,
                                          num_labels=num_labels, freeze_layers=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# Preprocess input text
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


# Predict class
def predict(model, tokenizer, text_pair, device, max_length=256):
    inputs = preprocess_text(text_pair, tokenizer, max_length)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)[1]
        _, prediction = torch.max(logits, dim=1)
    return prediction.cpu().item()


# Streamlit App
def main():
    st.set_page_config(page_title="Text Pair Classification", layout="centered", initial_sidebar_state="auto",
                       theme={"primaryColor": "#1F77B4", "backgroundColor": "#111", "secondaryBackgroundColor": "#222",
                              "textColor": "#fff"})

    st.title("Text Pair Classification")
    st.markdown("This app uses a fine-tuned BERT model to classify text pairs.")

    # Load tokenizer and model
    st.sidebar.write("Loading model...")
    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL_NAME)
    model = load_model(MODEL_PATH, PRETRAINED_MODEL_NAME, DEVICE, NUM_LABELS)
    st.sidebar.success("Model loaded successfully!")

    # Text inputs
    st.markdown("### Enter the Text Pair")
    text1 = st.text_input("First Text", value="")
    text2 = st.text_input("Second Text", value="")

    # Predict
    if st.button("Classify"):
        if not text1 or not text2:
            st.error("Both text inputs are required!")
        else:
            text_pair = (text1, text2)
            prediction = predict(model, tokenizer, text_pair, DEVICE, MAX_LENGTH)
            st.success(f"Predicted Class: {prediction}")

    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. Provide the text pair to classify in the text inputs.
    2. Click on the **Classify** button to see the prediction.
    """)


if __name__ == "__main__":
    main()
