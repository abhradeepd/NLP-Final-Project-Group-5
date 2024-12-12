Quora Question Pair Similarity Project
This project aims to classify question pairs from the Quora Question Pairs dataset as duplicate or non-duplicate, leveraging advanced NLP techniques and machine learning models. With over 400,000 question pairs, the dataset poses challenges due to its imbalance, with only 36.92% labeled as duplicates.

Methodology
Exploratory Data Analysis (EDA): Conducted statistical analyses and visualizations to understand data distribution and detect anomalies.
Feature Engineering:
Generated 37 features across fuzzy similarity metrics, token-based features, and common subsequence features.
Extracted BERT-based CLS embeddings for each question using a custom PyTorch DataLoader.
Classical Machine Learning Models:
Implemented Naïve Bayes, Logistic Regression, Random Forest, and XGBoost using a dynamic pipeline.
XGBoost achieved the highest F1-macro score of 84.10%.
Deep Learning Models:
Developed LSTM and GRU networks with shared fully connected layers to process embeddings and additional features.
Despite extensive tuning, these models underperformed, with losses oscillating between 0.65 and 0.69.
Transformer Models:
Implemented a modified Siamese Sentence BERT (SBERT) framework with fine-tuning of the last two encoder layers.
Used mean pooling as the most effective strategy and added two fully connected layers for prediction.
Achieved an F1-macro score of 89.07%, outperforming all other approaches.
Streamlit App: Developed an interactive app to classify question pairs and visualize trends, making the results accessible to users.
Conclusion
This project highlights the superior performance of transformer architectures like SBERT for semantic similarity tasks. Selective fine-tuning, effective pooling strategies, and careful monitoring of training epochs were crucial in achieving high accuracy and generalization.
