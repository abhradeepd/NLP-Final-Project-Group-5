import streamlit as st
import torch
from transformers import BertTokenizerFast
from SentencePair_similarity_Bert.model import BertForTextPairClassification

from ml_algorithms.tse_forStreamlit import plot_tsne_visualization
from ml_algorithms.tsne_for_question_pairs import load_data, TextPairDataset, get_embeddings,plot_embeddings_tsne_3d
from ml_algorithms.tsne_for_single_embeddings import (
    load_unique_questions,
    UniqueQuestionsDataset,
    extract_embeddings,
    plot_unique_questions_tsne_3d_with_clusters
)
from ml_algorithms.tsne_for_single_embeddings import SiameseBertModel
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import pandas as pd

# Predefined constants
filename = 'Data/processed_features.csv'
MODEL_PATH = "/Users/richikghosh/Documents/Quora Duplicate/SentencePair_similarity_Bert/best_model_state_bert_pair_mlp2.pt"
MODEL_PATH2 = "/Users/richikghosh/Documents/Quora Duplicate/Bert_contrastive/siamese_bert_model_best_contrastiveLoss_3lay.pth"
PRETRAINED_MODEL_NAME = 'bert-base-uncased'
DATA_FILE = 'Data/val_QQP_test.csv'
MAX_LENGTH = 256
NUM_LABELS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the model
@st.cache_resource
def load_model(model_path, pretrained_model_name, device, num_labels):
    model = BertForTextPairClassification(
        pretrained_model_name=pretrained_model_name, drop_out=0.1, num_labels=num_labels, freeze_layers=10
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.to(device)
    model.eval()
    return model
@st.cache_resource
def load_bert_model(model_path, pretrained_model_name, device, num_labels):
    model = SiameseBertModel(
        bert_model_name=pretrained_model_name)
    #drop_out=0.1, num_labels=num_labels

    model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
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
    # Set Page Configuration
    st.set_page_config(page_title="Text Pair Classification", layout="wide", initial_sidebar_state="expanded")

    # Sidebar Configuration
    with st.sidebar:
        st.title("🔧 Settings")
        st.markdown("---")
        with st.expander("Visualization Settings"):
            st.markdown("Configure parameters for t-SNE visualizations.")
            subset_size = st.slider(
                "Number of instances for t-SNE visualization",
                min_value=40,
                max_value=5000,
                step=10,
                value=100
            )
            iterations = st.slider(
                "t-SNE Iterations",
                min_value=250,
                max_value=4000,
                step=250,
                value=1000
            )
            perplexity = st.slider(
                "t-SNE Perplexity",
                min_value=5,
                max_value=50,
                step=5,
                value=30,
                help="Lower values focus on local structure, higher values on global structure."
            )
        st.markdown("---")
        st.write("Loading models...")
        tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL_NAME)
        model = load_model(MODEL_PATH, PRETRAINED_MODEL_NAME, DEVICE, NUM_LABELS)
        model2 = load_bert_model(MODEL_PATH2, PRETRAINED_MODEL_NAME, DEVICE, NUM_LABELS)
        st.success("Models loaded successfully!")

    # Main Section with Tabs
    st.title("Text Pair Classification 📚")
    st.markdown("Use a fine-tuned BERT model to classify the similarity of text pairs.")

    tab1, tab2, tab3 = st.tabs(["📊 Visualizations", "🔍 Classification", "ℹ️ Instructions"])

    # Visualization Tab
    with tab1:
        st.header("📊 Data Visualization")
        st.markdown("Explore processed features or embeddings using t-SNE visualizations.")

        # Load Processed Features
        st.subheader("Feature Space t-SNE Visualization")
        if os.path.isfile(filename):
            try:
                data_features = pd.read_csv(filename)
            except Exception as e:
                st.error(f"Error processing the file: {str(e)}")
                data_features = None
        else:
            st.error("Processed feature file not found. Please check the file path.")
            data_features = None

        if data_features is not None:
            st.markdown("#### Feature Space t-SNE Visualization (Processed Features)")
            if st.button("Generate 3D t-SNE Plot (Processed Features)"):
                tsne_3d_fig = plot_tsne_visualization(data_features, n_iter=iterations, n_components=3,
                                                      subset_size=subset_size, perplexity=perplexity)
                st.plotly_chart(tsne_3d_fig, use_container_width=True)

        st.subheader("🔎 Embeddings-based t-SNE Visualization")
        if st.button("Generate 3D t-SNE Plot (Embeddings)"):
            df = load_data(DATA_FILE)
            if subset_size < len(df):
                df = df.head(subset_size)

            dataset = TextPairDataset(df, tokenizer, max_length=MAX_LENGTH)
            data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

            embeddings, labels = get_embeddings(model, data_loader, DEVICE)
            q1 = df['question1'].tolist()[:len(embeddings)]
            q2 = df['question2'].tolist()[:len(embeddings)]

            tsne_3d_fig = plot_embeddings_tsne_3d(embeddings, labels, q1, q2, n_iter=iterations, perplexity=perplexity)
            st.plotly_chart(tsne_3d_fig, use_container_width=True)

        st.subheader("🌟 Unique Questions t-SNE Visualization")
        if st.button("Generate 3D t-SNE Plot (Unique Questions)"):
            unique_questions, unique_labels = load_unique_questions(DATA_FILE)
            if subset_size < len(unique_questions):
                unique_questions = unique_questions[:subset_size]
                unique_labels = unique_labels[:subset_size]

            unique_dataset = UniqueQuestionsDataset(
                questions=unique_questions,
                labels=unique_labels,
                tokenizer=tokenizer,
                max_len=MAX_LENGTH,
            )
            unique_dataloader = DataLoader(unique_dataset, batch_size=32, shuffle=False)

            unique_embeddings, unique_labels_array, unique_questions_array = extract_embeddings(
                model2, unique_dataloader, DEVICE
            )
            unique_tsne_fig = plot_unique_questions_tsne_3d_with_clusters(
                embeddings=unique_embeddings,
                questions=unique_questions_array,
                n_iter=iterations,
                perplexity=perplexity
            )
            st.plotly_chart(unique_tsne_fig, use_container_width=True)

    # Classification Tab
    with tab2:
        st.header("🔍 Text Pair Classification")
        st.markdown("Provide two text inputs for similarity classification.")
        col1, col2 = st.columns(2)
        with col1:
            text1 = st.text_input("Enter the first text:")
        with col2:
            text2 = st.text_input("Enter the second text:")
        if st.button("Classify"):
            if not text1 or not text2:
                st.error("Please provide both text inputs.")
            else:
                text_pair = (text1, text2)
                prediction = predict(model, tokenizer, text_pair, DEVICE, MAX_LENGTH)
                st.success(f"Prediction: **Class {prediction}**")

    # Instructions Tab
    with tab3:
        st.header("ℹ️ Instructions")
        st.markdown("""
        ### How to Use:
        1. **Enter Text Pair:** Go to the "Classification" tab, provide two text inputs, and press "Classify".
        2. **Visualize Processed Features:** In the "Visualizations" tab, press "Generate 3D t-SNE Plot (Processed Features)".
        3. **Visualize Embeddings:** Explore embeddings for duplicates and non-duplicates using the appropriate buttons.
        """)
        st.markdown("---")



if __name__ == "__main__":
    main()
