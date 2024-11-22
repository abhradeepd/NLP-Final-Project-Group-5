from pre_processing import preprocess
from nltk.corpus import stopwords
import os
import pandas as pd
import distance
from fuzzywuzzy import fuzz


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
        data['freq_qid1'] = data.groupby('qid1')['qid1'].transform('count') # frequency of qids
        data['freq_qid2'] = data.groupby('qid2')['qid2'].transform('count')
        data['q1len'] = data['question1'].str.len() # number character of the questions
        data['q2len'] = data['question2'].str.len()
        data['q1_n_words'] = data['question1'].apply(lambda row: len(row.split(" "))) # number of words
        data['q2_n_words'] = data['question2'].apply(lambda row: len(row.split(" ")))




        data['fre_q1-q2'] = abs(data['freq_qid1'] - data['freq_qid2'])
        data['freq_q1+q2'] = data['freq_qid1'] + data['freq_qid2']

        def normalized_word_Common(row):
            w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
            w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
            return 1.0 * len(w1 & w2) # for intersection

        data['word_Common'] = data.apply(normalized_word_Common, axis=1)
        def normalized_word_Total(row):
            w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
            w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
            return 1.0 * (len(w1)+len(w2))
        data['word_Total'] = data.apply(normalized_word_Total,axis=1)

        def normalized_word_share(row):
            w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
            w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
            return 1.0 * len(w1 & w2) / (len(w1) + len(w2))  # [common words / total words]

        data['word_share'] = data.apply(normalized_word_share, axis=1)

        def get_longest_substr_ratio(a, b):
            strs = list(distance.lcsubstrings(a, b))
            if len(strs) == 0:
                return 0
            else:
                return len(strs[0]) / (min(len(a), len(b)) + 1) # [lenght_of_substring/ (smaller_string+1)]

        def longest_common_subsequence(q1, q2):
            # Function to calculate Longest Common Subsequence
            seq1 = list(q1)
            seq2 = list(q2)

            m, n = len(seq1), len(seq2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if seq1[i - 1] == seq2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            lenght_lcs= dp[m][n]

            return lenght_lcs/(max(len(seq1),len(seq2))+1) # Return noramlized common subsequnce length

        #Preprocessing
        data['question1'] = data['question1'].fillna("").apply(preprocess)
        data['question2'] = data['question2'].fillna("").apply(preprocess)

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

        print("fuzzy features..")
        # Compare sorted token and removes duplicates then similarity score (** Ignores unmatched words)
        data["token_set_ratio"] = data.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
        # Sort the questions alphabetically and then check similarity score (** Penalizes for unmatched words)
        data["token_sort_ratio"] = data.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
        # Here a word to word comparison is done no sorting so (orange and apples and apples and oranges will score low) Here the edit distance is used to find similarity of tokens
        data["fuzz_ratio"] = data.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
        # This checks if one string in present inside another larger string (e.g. Jaipur pink city and Jaipur)  Here the edit distance is used to find similarity of tokens
        data["fuzz_partial_ratio"] = data.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
        # This is a harsher substring matcher than partial ratio as partial one accepts appx. match
        data["longest_substr_ratio"] = data.apply(lambda x: get_longest_substr_ratio(x["question1"], x["question2"]),
                                                  axis=1)

        print("Common Subsequence")
        # Finds the largest common subsequence
        data["largest_common_subsequence"] = data.apply(lambda x: longest_common_subsequence(x["question1"], x["question2"]),axis=1)
    return data


k=process_file_and_extract_features('train.csv',5)
print(k)

