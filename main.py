#%%
import os 
import sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

#%%
data = pd.read_csv('data/train.csv')

#%%
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
# Function to count unique words in a column
def count_unique_words(column):
    all_words = ' '.join(column).split()
    unique_word_counts = Counter(all_words)
    return unique_word_counts

# Count unique words in each column
q1_unique_words = count_unique_words(data['question1'])
q2_unique_words = count_unique_words(data['question2'])

# Convert Counter to DataFrame for easier manipulation and visualization
q1_df = pd.DataFrame(q1_unique_words.items(), columns=['word', 'frequency_question1']).sort_values(by='frequency_question1', ascending=False).head(50)
q2_df = pd.DataFrame(q2_unique_words.items(), columns=['word', 'frequency_question2']).sort_values(by='frequency_question2', ascending=False).head(50)

# Merging the two DataFrames for comparison
merged_df = pd.merge(q1_df, q2_df, on='word', how='outer').fillna(0)

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
# %%
