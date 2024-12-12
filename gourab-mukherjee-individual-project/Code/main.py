#%%
import os 
import sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
from pre_processing import preprocess
#%%
from ml_algorithms.tSNE_for_data_visualization import plot_tsne_visualization
 #%%
from feature_extraction import process_file_and_extract_features
from quiz import train

#%%
data = pd.read_csv('Data/train.csv')

#%%
pd.set_option('display.max_rows', None)

# Display all columns
pd.set_option('display.max_columns', None)

data.head(5)

# %%
print(f"Number of Observations in data are {data.shape[0]}")

# %%
print(data.info())

missing_values_count = data.isnull().sum()

# Create a bar plot of missing values
plt.figure(figsize=(10, 6))
missing_values_count.plot(kind='bar', color='skyblue')

# Add labels and title
plt.title('Number of Missing Values per Column', fontsize=16)
plt.xlabel('Columns', fontsize=14)
plt.ylabel('Number of Missing Values', fontsize=14)

# Show the plot
plt.show()

#%%
# Count the number of rows before dropping
rows_before_drop = len(data)

# Drop rows with missing values
data = data.dropna()

# Count the number of rows after dropping
rows_after_drop = len(data)

# Calculate the number of rows dropped
rows_dropped = rows_before_drop - rows_after_drop

# Display the number of rows dropped
print("Number of rows dropped:", rows_dropped)

#%%
# Group by 'is_duplicate' and count the number of observations for each group
grouped_data = data.groupby("is_duplicate")['id'].count()

total_questions = grouped_data.sum()
percentages = (grouped_data / total_questions) * 100

colors = ['lightblue', 'lightcoral']  
plt.figure(figsize=(10, 8))  
ax = percentages.plot(kind='bar', color=colors, edgecolor='black')

plt.title('Distribution of Duplicate and Non-duplicate Questions', fontsize=16)
plt.xlabel('is_duplicate', fontsize=14)
plt.ylabel('Percentage of Questions', fontsize=14)

for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=12)

ax.set_xticklabels(['Non-duplicate', 'Duplicate'], rotation=0)

plt.show()

#%%
qids = pd.Series(data['qid1'].tolist() + data['qid2'].tolist())
unique_qs = len(np.unique(qids))
qs_morethan_onetime = np.sum(qids.value_counts() > 1)

print('Total number of Unique Questions: {}\n'.format(unique_qs))
print('Number of unique questions that appear more than one time: {} ({}%)\n'.format(qs_morethan_onetime, round(qs_morethan_onetime/unique_qs*100,2)))
print('Max number of times a single question is repeated: {}\n'.format(max(qids.value_counts())))

#%%
all_questions = pd.concat([data['question1'], data['question2']], ignore_index=True)

# Display the top 10 most common questions
top_10_common_questions = all_questions.value_counts().head(10)

print("Top 10 Most Common Questions:")
print(top_10_common_questions)

#%%
# Plot the number of unique questions and repeated questions
plt.figure(figsize=(8, 6))
colors = ['yellow', 'lightgreen']
plt.bar(['Unique Questions', 'Repeated Questions'], [unique_qs, qs_morethan_onetime], color=colors, edgecolor='black')

plt.title('Number of Unique and Repeated Questions', fontsize=16)
plt.ylabel('Number of Questions', fontsize=14)

# Add text annotations
for i, count in enumerate([unique_qs, qs_morethan_onetime]):
    plt.text(i, count + 0.1, str(count), ha='center', va='bottom', fontsize=12)

plt.show()

#%%
plt.figure(figsize=(20, 10))

counts, bins, _ = plt.hist(qids.value_counts(), bins=160, color='skyblue', edgecolor='black')

plt.yscale('log', nonpositive='clip')

plt.title('Log-Histogram of question appearance counts', fontsize=16)
plt.xlabel('Number of occurrences of question', fontsize=14)
plt.ylabel('Number of questions', fontsize=14)

max_occurrence = max(qids.value_counts())
plt.axvline(x=max_occurrence, color='red', linestyle='--', label=f'Max Occurrence: {max_occurrence}')

plt.legend()

plt.show()

#%%
def count_words(sentence):
    # Handle the case where the sentence is NaN (missing value)
    if pd.isnull(sentence):
        return 0
    # Count the number of words by splitting the sentence
    return len(str(sentence).split())

# Plot histograms for question lengths
plt.figure(figsize=(12, 6))
plt.hist(data['question1'].apply(lambda x: count_words(x)), bins=50, alpha=0.5, label='Question 1', color='blue')
plt.hist(data['question2'].apply(lambda x: count_words(x)), bins=50, alpha=0.5, label='Question 2', color='orange')

# Title and labels
plt.title('Distribution of Question Lengths', fontsize=16)
plt.xlabel('Number of words', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

# Display legend
plt.legend()

plt.show()

#%%
# Ensure you have the stopwords downloaded
nltk.download('stopwords')

# Constants
SAFE_DIV = 0.0001
STOP_WORDS = set(stopwords.words("english"))

# Apply preprocessing to each column
data['question1'] = data['question1'].apply(preprocess)
data['question2'] = data['question2'].apply(preprocess)

# Define the function to count unique words excluding stopwords
def count_unique_words(column):
  stop_words = STOP_WORDS
  all_words = ' '.join(column).split()
  filtered_words = [word for word in all_words if word.lower() not in stop_words]
  unique_word_counts = Counter(filtered_words)
  return unique_word_counts

# Count unique words in each column
q1_unique_words = count_unique_words(data['question1'])
q2_unique_words = count_unique_words(data['question2'])

# Convert Counter to DataFrame for easier manipulation and visualization
q1_df = pd.DataFrame(q1_unique_words.items(), columns=['word', 'frequency_question1'])
q2_df = pd.DataFrame(q2_unique_words.items(), columns=['word', 'frequency_question2'])

# Merging the two DataFrames for comparison
merged_df = pd.merge(q1_df, q2_df, on='word', how='outer').fillna(0)

# Add total frequency column and sort by it
merged_df['total_frequency'] = merged_df['frequency_question1'] + merged_df['frequency_question2']
merged_df = merged_df.sort_values(by='total_frequency', ascending=False).head(50)

# Plotting the data
plt.figure(figsize=(14, 8))

# Creating a grouped bar plot
bar_width = 0.35
index = range(len(merged_df))

plt.bar(index, merged_df['frequency_question1'], bar_width, label='Question 1', color='blue', alpha=0.6)
plt.bar([i + bar_width for i in index], merged_df['frequency_question2'], bar_width, label='Question 2', color='red', alpha=0.6)

plt.xlabel('Unique Words')
plt.ylabel('Frequency')
plt.title('Top 50 Unique Word Frequency Comparison between Question 1 and Question 2')
plt.xticks([i + bar_width / 2 for i in index], merged_df['word'], rotation=90)
plt.legend()

# Adjust layout to prevent label cutoff
plt.tight_layout()

plt.show()

# %%
# Function to count common words between question pairs
def count_common_words(row):
    q1_words = set(row['question1'].split())
    q2_words = set(row['question2'].split())
    return len(q1_words.intersection(q2_words))

# Assuming 'data' is your DataFrame with 'question1', 'question2', and 'is_duplicate' columns

# Creating 'common_words' column
data['common_words'] = data.apply(count_common_words, axis=1)

# Plotting the distribution of common words
plt.figure(figsize=(14, 6))

# KDE plot for duplicate questions (blue) with adjustments
sns.kdeplot(data[data['is_duplicate'] == 1]['common_words'], 
            shade=True, color='blue', label='Similar Questions', bw_adjust=2.0, alpha=0.6) 

# KDE plot for non-duplicate questions (red) with adjustments
sns.kdeplot(data[data['is_duplicate'] == 0]['common_words'], 
            shade=True, color='red', label='Non-Similar Questions', bw_adjust=2.0, alpha=0.6) 

# Setting the title and labels
plt.title('Distribution of Common Words Between Questions')
plt.xlabel('Number of Common Words')
plt.ylabel('Density')

# Setting the x-axis range to 0 to 15
plt.xlim(0, 15)

# Adjusting x-axis ticks
plt.xticks([0, 2.5, 5, 7.5, 10, 12.5, 15])

# Adding legend
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()
#%%
len(data)
# %%
import importlib
import feature_extraction
importlib.reload(feature_extraction)
from feature_extraction import process_file_and_extract_features
#%%

filename = 'Data/train.csv'
# filename = r'/home/ubuntu/NLP-Final_Project-Group-5/NLP-Final-Project-Group-5/Data/train.csv'
if os.path.isfile(filename):
    data_features = process_file_and_extract_features(filename,len(data)-1)
else:
    print("File not found")
#%%
data_features.columns
#%%
import visualise
# importlib.reload(visualise)
#Fucntion to vislaise violin and density plots for each feature in one image
from visualise import violin_density_plot_each_feature,kl_divergence_visualise,plot_for_top_5_features
#%%
features_to_plot = ['freq_qid1', 'freq_qid2', 'q1len', 'q2len', 'q1_n_words', 'q2_n_words',
                    'word_Common', 'word_Total', 'word_share', 'freq_q1+q2', 'freq_q1-q2',
                    'ratio_q_lengths', 'common_prefix', 'common_suffix', 'diff_words', 'diff_chars',
                    'jaccard_similarity', 'largest_common_subsequence', 'cwc_min', 'cwc_max', 'csc_min',
                    'csc_max', 'ctc_min', 'ctc_max', 'last_word_eq', 'first_word_eq', 'abs_len_diff',
                    'mean_len', 'token_set_ratio', 'token_sort_ratio', 'fuzz_ratio', 'fuzz_partial_ratio',
                    'longest_substr_ratio']


#%%
violin_density_plot_each_feature(data_features,features_to_plot)
#%%
data_features.columns

# %%
kl_divergence_results = kl_divergence_visualise(data_features,features_to_plot)
#%%
kl_divergence_results


# %%
plot_for_top_5_features(data_features,kl_divergence_results)
#%%
data_features.columns
#%%
data_features.to_csv('data_with_features.csv')
#%%

# %%
from importlib import reload
import ml_algorithms.tSNE_for_data_visualization  # Import the module
reload(ml_algorithms.tSNE_for_data_visualization)  # Force reload the module
from ml_algorithms.tSNE_for_data_visualization import plot_tsne_visualization  # Import the specific function
#%%
plot_tsne_visualization(data_features)

# %%
######
data_features.shape
#%%
df = pd.read_csv('Data/train.csv')
df.head(4)
#%%
import importlib
import tfidf_new
importlib.reload(tfidf_new)
#%%
data_features = pd.read_csv('data_with_features.csv')
#%%
data_features.columns
# df.shape
#%%
from tfidf_new import tfidf_calculate,final_feature_creation
df_with_tfidf_vectors = tfidf_calculate(df)
#%%
df_with_tfidf_vectors.columns
#%%
df_with_tfidf_vectors.to_csv('tfidf_vectors_genrated_new.csv')
#%%
final_features = final_feature_creation(data_features,df_with_tfidf_vectors)
final_features.shape
#%%
final_features.shape
#%%
final_features.to_csv('final_features_generated_new.csv')
#%%
final_features = pd.read_csv('final_features_generated_new.csv')
final_features.shape
#%%
# print(f'from final features of tfidf {final_features.shape}')
# print(f'reading final df from saved final features {final_df.shape}')

#%%
# Replace non-numeric values with NaN
# final_df.replace({col: {'_x': np.nan} for col in data.columns}, inplace=True)

# Check if there are any NA values in the DataFrame
if final_features.isna().any().any():
    print("NA Values Present")
else:
    print("No NA Values Present")

# Check the number of NaN values in each column after replacement
nan_counts = final_features.isna().sum()
print("Number of NaN values in each column after replacement:")
print(nan_counts)
#%%
final_df_ex = final_features.copy(deep=True)
final_df_ex.shape
#%%
final_df_ex.columns
#%%
# Get the target variable
y_true = final_df_ex['is_duplicate']
final_df_ex.drop(['Unnamed: 0.1','Unnamed: 0','id', 'is_duplicate'], axis=1, inplace=True)
print(f'final df shape {final_df_ex.shape}')

#%%

# Convert all the features into numeric
cols = list(final_df_ex.columns)
for i in cols:
    final_df_ex[i] = pd.to_numeric(final_df_ex[i], errors='coerce')

print(final_df_ex.shape)
print(final_df_ex.head())
#%%
# Check for NA Values
if final_df_ex.isna().any().any():
    print("NA Values Present")
else:
    print("No NA Values Present")
# Check the number of NaN values in each column after conversion
nan_counts_after_conversion = final_df_ex.isna().sum()
print("Number of NaN values in each column after conversion:")
print(nan_counts_after_conversion)

# Convert y_true to a list of integers
y_true = list(map(int, y_true.values))

# Display the first few rows of the data
final_df_ex.head()
#%%
from train_test_split_and_random_model import splitting,distribution_outputvariable_train_test
X_train, X_test, y_train, y_test = splitting(final_df_ex,y_true,test_size=0.3)
#%%
distribution_outputvariable_train_test(y_train,y_test)
#%%
import importlib
import train_test_split_and_random_model
importlib.reload(train_test_split_and_random_model)
from train_test_split_and_random_model import random_model
random_model(y_train,y_test)
#%%
from importlib import reload
import ml_algorithms.Logistic_Regression  # Import the module
reload(ml_algorithms.Logistic_Regression)  # Force reload the module
from ml_algorithms.Logistic_Regression import logistic_regression_function
#%%
logistic_regression_function(X_train,X_test,y_train,y_test)
#%%
import ml_algorithms.SGDClassfier_RandomSearch  # Import the module
reload(ml_algorithms.SGDClassfier_RandomSearch)  # Force reload the module
from ml_algorithms.SGDClassfier_RandomSearch import sgd_random_search_v1
sgd_random_search_v1(X_train,X_test,y_train.values.ravel(), y_test.values.ravel())
#%%
import ml_algorithms.NaiveBayes
reload(ml_algorithms.NaiveBayes)
from ml_algorithms.NaiveBayes import naive_bayes_function
#%%
naive_bayes_function(X_train,X_test,y_train, y_test)
#%%
import ml_algorithms.XGBoost_Random_Search
reload(ml_algorithms.XGBoost_Random_Search)
from ml_algorithms.XGBoost_Random_Search import xgboost_random_search
#%%
xgboost_random_search(X_train,X_test,y_train,y_test)
#%%
from ml_algorithms.RandomForest import random_forest_random_search
random_forest_random_search(X_train,X_test,y_train,y_test)