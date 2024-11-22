import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Creates one image with Violin plots and Density plots for each feature
def violin_density_plot_each_feature(data,features_to_plot):
    print("updates")
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
    