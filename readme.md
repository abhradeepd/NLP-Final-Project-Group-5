# Quora Question Pair Similarity Classification

## Members
- **Abhradeep Das** (G38747350)
- **Gourab Mukherjee** (G32241729)
- **Richik Ghosh** (G31267506)
- **Shreya Sahay** (G36642286)

---

## Introduction

Quora is a dynamic platform for acquiring and disseminating knowledge across a broad spectrum of topics, attracting over 100 million visitors monthly. However, its immense popularity has led to a proliferation of similarly phrased questions, making it challenging for users to find the most relevant answers and for contributors to avoid redundancy.

This project focuses on developing a system to classify whether pairs of questions are contextually similar using the **Quora Question Pairs dataset**. We employed classical machine learning models, neural network-based approaches, and state-of-the-art transformer-based models to tackle this problem. The goal is to:

1. Streamline the user experience.
2. Enhance Quora's infrastructure by reducing redundancy.
3. Contribute to the broader field of Natural Language Processing (NLP).

---

## Shared Workflow

### 1. Exploratory Data Analysis (EDA)
- Analyzed data distribution, identified patterns, and detected anomalies.
- Used visualizations and statistical summaries.

### 2. Data Preprocessing
- **Normalization**: Converted text to lowercase, expanded contractions, standardized symbols.
- **Cleaning**: Removed HTML tags and non-word characters.
- **Stemming**: Applied the `PorterStemmer` algorithm.

### 3. Feature Extraction
- **Tokenization**: Split text into individual words.
- Extracted 37 features in three categories:
  - **Fuzzy Features**: e.g., `fuzz_ratio`, `fuzz_partial_ratio`.
  - **Token Features**: e.g., common stopwords and non-stopwords, word size difference.
  - **Common Subsequence Features**: e.g., `largest_common_subsequence`, `jaccard_similarity`.

### 4. Vectorization
- **Weighted TF-IDF**: Enhanced feature representation by combining TF-IDF scores with SpaCy embeddings.
- **BERT Embeddings**: Generated contextualized embeddings using pre-trained BERT models.

### 5. Model Training and Comparison
- Classical machine learning models (e.g., Logistic Regression, Random Forest).
- Deep learning models (e.g., LSTM, GRU).
- Transformer-based models (e.g., BERT variations).

### 6. Streamlit App
- Developed an interactive app to classify question pairs and visualize embeddings.

---

## Dataset Description

Sourced from a [Kaggle competition](https://www.kaggle.com/competitions/quora-question-pairs/data), the dataset contains:
- **404,000 question pairs** with 6 key features:
  - `id`: Identifier for the question pair.
  - `qid1`, `qid2`: Unique identifiers for each question.
  - `question1`, `question2`: The full text of the questions.
  - `is_duplicate`: Target variable (1 for duplicate, 0 for non-duplicate).
- Class imbalance: 63.08% non-duplicate, 36.92% duplicate.

---

## Methodology

### Data Preprocessing
- Transformed text to lowercase, expanded contractions, removed non-word characters, and stripped HTML tags.
- Applied stemming to reduce vocabulary size while retaining semantic meaning.

### Feature Extraction
- Extracted 37 features categorized into:
  - **Fuzzy Features**: e.g., `fuzz_ratio`, `fuzz_partial_ratio`.
  - **Token Features**: e.g., common stopwords and non-stopwords.
  - **Common Subsequence Features**: e.g., `largest_common_subsequence`.

### Vectorization
- Weighted TF-IDF scores combined with SpaCy embeddings.
- BERT embeddings captured contextual semantics.

---

## Modeling

### Classical Models
- Logistic Regression, Random Forest, Naïve Bayes, XGBoost.


### Deep Learning Models
- LSTM and GRU architectures using BERT embeddings.

### Transformer Models
- Variations of BERT, including Siamese BERT and Cross-Encoder architectures.

**ALL THE ARCHITECTURES ARE PRESENT IN THE GROUP REPORT RICHIK GHOSH AND SHREYA SAHAY'S PERSONAL REPORTS**

---

## Results

| Algorithm                        | F1-Macro Score |
|----------------------------------|----------------|
| Naïve Bayes (TF-IDF)             | 61.3%          |
| Logistic Regression (TF-IDF)     | 74.0%          |
| Random Forest (TF-IDF)           | 81.7%          |
| XGBoost (TF-IDF)                 | 83.0%          |
| Naïve Bayes (BERT)               | 61.82%         |
| Logistic Regression (BERT)       | 79.31%         |
| Random Forest (BERT)             | 82.04%         |
| XGBoost (BERT)                   | 84.10%         |
| GRU (BERT)                       | 38.85%         |
| LSTM (BERT)                      | 39.93%         |
| Siamese BERT + FCNN              | 89.07%         |
| Siamese BERT + Dynamic Masking   | 89.68%         |
| SBERT + Contrastive Loss         | 80.48%         |
| BERT + Contrastive Loss          | 79.48%         |
| Cross-Encoder BERT               | **93.28%**     |

### Summary
The **Cross-Encoder BERT model** achieved the best performance with an F1-macro score of **93.28%**, followed by Siamese BERT models. Classical models performed well but were outclassed by transformer-based approaches.

---

## Streamlit App

- Provides an interface to classify question similarity.
- Visualizes data with 2D and 3D t-SNE plots for embeddings.

---

## References
- [First Quora Dataset Release: Question Pairs](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)
- [Sentence-BERT](https://www.sbert.net/)
- Python libraries: pandas, NumPy, matplotlib, seaborn, NLTK.

---

## Repository Link

Find the complete code and resources in our [GitHub repository](https://github.com/abhradeepd/NLP-Final-Project-Group-5).
