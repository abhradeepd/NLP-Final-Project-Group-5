import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Creates one image with Violin plots and Density plots for each feature
def violin_density_plot_each_feature(data,features_to_plot):
    print("updates1")
    num_features = len(features_to_plot)
    plt.figure(figsize=(16, 2*num_features))

    for i, feature in enumerate(features_to_plot):
        plt.subplot(num_features, 2, 2*i + 1)
        sns.violinplot(x='is_duplicate', y=feature, data=data)
        plt.title(f'Violin Plot for {feature}')

        plt.subplot(num_features, 2, 2*i + 2)
        sns.kdeplot(data[data['is_duplicate'] == 1][feature], label='Duplicate', fill=True, warn_singular=False)
        sns.kdeplot(data[data['is_duplicate'] == 0][feature], label='Not Duplicate', fill=True, warn_singular=False)
        plt.title(f'Density Plot for {feature}')

        plt.tight_layout()
        plt.show()
    
    
    return



# Function to calculate KL Divergence
def calculate_kl_divergence(duplicate_data, non_duplicate_data, feature):
    duplicate_dist = duplicate_data[feature].dropna()
    non_duplicate_dist = non_duplicate_data[feature].dropna()

    epsilon = 1e-10
    duplicate_dist += epsilon
    non_duplicate_dist += epsilon

    min_length = min(len(duplicate_dist), len(non_duplicate_dist))
    duplicate_dist = duplicate_dist.head(min_length)
    non_duplicate_dist = non_duplicate_dist.head(min_length)

    kl_divergence = entropy(duplicate_dist, non_duplicate_dist)
    return kl_divergence



def kl_divergence_visualise(data,features_to_plot):
    
    kl_divergence_results = pd.DataFrame(columns=['Feature', 'KL_Divergence'])
    
    for feature in features_to_plot:
        kl_divergence = calculate_kl_divergence(data[data['is_duplicate'] == 1], data[data['is_duplicate'] == 0], feature)
        kl_divergence_results = pd.concat([kl_divergence_results, pd.DataFrame({
            'Feature': [feature],
            'KL_Divergence': [kl_divergence]
        })], ignore_index=True)

    # Display KL Divergence results in a table
    print(kl_divergence_results)

    # Create a bar plot to visualize inverted KL Divergence
    kl_divergence_results['Inverted_KL_Divergence'] = 1 / (kl_divergence_results['KL_Divergence'] + 1e-10)

    plt.figure(figsize=(15, 10))
    sns.barplot(x='Feature', y='Inverted_KL_Divergence', data=kl_divergence_results.sort_values(by='Inverted_KL_Divergence', ascending=False))
    plt.title('Inverted KL Divergence for Each Feature')
    plt.xticks(rotation=45, ha='right')
    plt.show()
    
    return kl_divergence_results




def plot_for_top_5_features(data,kl_divergence_results):
    bottom_5_features = kl_divergence_results.nsmallest(5, 'KL_Divergence')['Feature']

    print("The best 5 features are:")
    print(bottom_5_features)

    # Pair plot for the top 10 features
    n = data.shape[0]
    sns.pairplot(data[bottom_5_features.tolist() + ['is_duplicate']][0:n], hue='is_duplicate', vars=bottom_5_features.tolist())
    plt.show()
    
    return 
