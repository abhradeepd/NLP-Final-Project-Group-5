import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import warnings
import numpy as np
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings("ignore")
import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import spacy
from spacy.cli import download

def tfidf_calculate(data):
    df = data
    df['question1'] = df['question1'].apply(lambda x: str(x))
    df['question2'] = df['question2'].apply(lambda x: str(x))

    # Combine texts
    questions = list(df['question1']) + list(df['question2'])

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(lowercase=False)
    tfidf.fit_transform(questions)
    word2tfidf = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))

    def load_spacy_model(model_name='en_core_web_lg'):
        try:
            # Check if the spaCy model is already installed
            nlp = spacy.load(model_name)
            print(f"spaCy model '{model_name}' loaded successfully.")
        except OSError:
            print(f"Model '{model_name}' not found. Downloading...")
            # Download the spaCy model if not already installed
            download(model_name)
            nlp = spacy.load(model_name)
            print(f"spaCy model '{model_name}' downloaded and loaded successfully.")

        return nlp

    # Load spaCy model
    nlp = load_spacy_model()

    # Extract features using spaCy
    vecs1 = []
    vecs2 = []

    for qu1, qu2 in tqdm(zip(list(df['question1']), list(df['question2']))):
        doc1 = nlp(qu1)
        mean_vec1 = np.zeros([len(doc1), len(doc1[0].vector)])

        for word1 in doc1:
            vec1 = word1.vector
            try:
                idf = word2tfidf[str(word1)]
            except:
                idf = 0
            mean_vec1 += vec1 * idf

        mean_vec1 = mean_vec1.mean(axis=0)
        vecs1.append(mean_vec1)

        doc2 = nlp(qu2)
        mean_vec2 = np.zeros([len(doc2), len(doc2[0].vector)])

        for word2 in doc2:
            vec2 = word2.vector
            try:
                idf = word2tfidf[str(word2)]
            except:
                idf = 0
            mean_vec2 += vec2 * idf

        mean_vec2 = mean_vec2.mean(axis=0)
        vecs2.append(mean_vec2)

    df['q1_feats_m'] = list(vecs1)
    df['q2_feats_m'] = list(vecs2)
    return df


def final_feature_creation(data_with_features,df_with_tfidf_features):
    dfnlp = data_with_features
    df = df_with_tfidf_features
    df1 = dfnlp.drop(['qid1', 'qid2', 'question1', 'question2'], axis=1)
    df3 = df.drop(['qid1', 'qid2', 'question1', 'question2', 'is_duplicate'], axis=1)
    df3_q1 = pd.DataFrame(df3.q1_feats_m.values.tolist(), index=df3.index)
    df3_q2 = pd.DataFrame(df3.q2_feats_m.values.tolist(), index=df3.index)
    # Display information about features
    print("Number of features in nlp dataframe:", df1.shape[1])
    print("Head(5) of nlp dataframe:")
    print(df1.head(5))

    print("\nNumber of features in question1 w2v dataframe:", df3_q1.shape[1])
    print("Head(5) of question1 w2v dataframe:")
    print(df3_q1.head(5))

    print("\nNumber of features in question2 w2v dataframe:", df3_q2.shape[1])
    print("Head(5) of question2 w2v dataframe:")
    print(df3_q2.head(5))

    print("\nNumber of features in the final dataframe:", df1.shape[1] + df3_q1.shape[1] + df3_q2.shape[1])

    df3_q1['id'] = df1['id']
    df3_q2['id'] = df1['id']

    # Merge df1 with df3_q1 and df3_q2 on 'id'
    result = df1.merge(df3_q1, on='id', how='left').merge(df3_q2, on='id', how='left')

    return result
