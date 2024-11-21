from pre_processing import preprocess
from nltk.corpus import stopwords
import os
import pandas as pd

stopwords = stopwords.words('english')

DIV_ERROR = .000001

def extract_basic_features(q1,q2):
    feature_tokens = [0.0] * 14
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return feature_tokens
    q1_non_stop_words = set([words for words in q1_tokens if words not in stopwords])
    q2_non_stop_words = set([words for words in q2_tokens if words not in stopwords])

    q1_stopwords = set([words for words in q1_tokens if words in stopwords])
    q2_stopwords = set([words for words in q2_tokens if words in stopwords])

    # common word count
    common_word_len = len(q1_non_stop_words.intersection(q2_non_stop_words))

    # common stopword count
    common_stop_word_len = len(q1_stopwords.intersection(q2_stopwords))

    # common token count
    common_token = len(set(q1_tokens).intersection(set(q2_tokens)))

    q1_non_stop_words = list(q1_non_stop_words)
    q2_non_stop_words = list(q2_non_stop_words)

    # Features based on the size of the questions
    feature_tokens[0] = common_word_len / (min(len(q1_non_stop_words), len(q2_non_stop_words)) + DIV_ERROR) # common_minimum_words
    feature_tokens[1] = common_word_len / (max(len(q1_non_stop_words), len(q2_non_stop_words))+DIV_ERROR) # common_maximum_word
    feature_tokens[2] = common_stop_word_len / (max(len(q1_stopwords), len(q2_stopwords))+DIV_ERROR) # common stopword max
    feature_tokens[3] = common_stop_word_len / (min(len(q1_stopwords), len(q2_stopwords)) + DIV_ERROR) # common stopword min
    feature_tokens[4] = common_token / (max(len(q1_tokens), len(q2_tokens))+DIV_ERROR) # common token max
    feature_tokens[5] = common_token / (min(len(q1_tokens), len(q2_tokens)) + DIV_ERROR) # common token min
    feature_tokens[6] = int(q2_tokens[0] == q1_tokens[0]) # common first token
    feature_tokens[7] = int(q2_tokens[-1] == q1_tokens[-1]) # common_last_token
    feature_tokens[8] = abs(len(q1_tokens) - len(q2_tokens)) #abs token size diff
    feature_tokens[9] = (len(set(q1_tokens)) + len(set(q2_tokens)))/2 # mean token size
    #non stop words
    feature_tokens[10] = int(q1_non_stop_words[0] == q2_non_stop_words[0])  # common first words
    feature_tokens[11] = int(q1_non_stop_words[-1] == q2_non_stop_words[-1])  # common_last_words
    feature_tokens[12] = abs(len(q1_non_stop_words) - len(q2_non_stop_words))  # abs words size diff
    feature_tokens[13] = (len(set(q1_non_stop_words)) + len(set(q2_non_stop_words))) / 2  # mean words size

    return feature_tokens


# q1, q2 = "Which one dissolve in water quikly sugar, salt, methane and carbon di oxide?","Which fish would survive in salt water?"
#
# # Check
# print(extract_basic_features(q1,q2))

def process_file_and_extract_features(filename, rows_to_train):
    if os.path.isfile(filename):
        data = pd.read_csv(filename,encoding="utf-8")
    else:
        data = pd.read_csv('Data/train.csv')
        data = data[:rows_to_train]
        # count of qids of question pairs
        data.dropna(subset=['question1','question2'],inplace=True)
        data['fre_qid1'] = data.groupby('qid1')['qid1'].transform('count') # frequency of qids
        data['freq_qid2'] = data.groupby('qid2')['qid2'].transform('count')
        data['q1len'] = data['question1'].str.len() # number character of the questions
        data['q2len'] = data['question2'].str.len()
        data['q1_n_words'] = data['question1'].apply(lambda row: len(row.split(" "))) # number of words
        data['q2_n_words'] = data['question2'].apply(lambda row: len(row.split(" ")))
        print("token features...")

        # Merging Features with dataset
        token_features = data.apply(lambda x: extract_basic_features(x["question1"], x["question2"]), axis=1)

        data["cwc_min"] = list(map(lambda x: x[0], token_features))
        data["cwc_max"] = list(map(lambda x: x[1], token_features))
        data["csc_min"] = list(map(lambda x: x[2], token_features))
        data["csc_max"] = list(map(lambda x: x[3], token_features))
        data["ctc_min"] = list(map(lambda x: x[4], token_features))
        data["ctc_max"] = list(map(lambda x: x[5], token_features))
        data["last_word_eq"] = list(map(lambda x: x[6], token_features))
        data["first_word_eq"] = list(map(lambda x: x[7], token_features))
        data["abs_len_diff"] = list(map(lambda x: x[8], token_features))
        data["mean_len"] = list(map(lambda x: x[9], token_features))
        data["first_word_eq_cw"] = list(map(lambda x: x[10], token_features))
        data["last_word_eq_cw"] = list(map(lambda x: x[11], token_features))
        data["abs_len_diff_cw"] = list(map(lambda x: x[12], token_features))
        data["mean_len_cw"] = list(map(lambda x: x[13], token_features))
    return data


k=process_file_and_extract_features('train.csv',5)
print(k)

