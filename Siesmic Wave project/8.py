# Analytics report

import numpy as np1
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load features from CSV
def load_features(features_file):
    features = pd.read_csv(features_file)
    return features

# Visualize feature distributions
def plot_feature_distributions(features):
    plt.figure(figsize=(14, 10))
    num_features = features.shape[1]
    num_rows = (num_features // 3) + 1  # Adjust layout for number of features
    
    for i, column in enumerate(features.columns):
        plt.subplot(num_rows, 3, i + 1)
        sns.histplot(features[column], kde=True)
        plt.title(column)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

# Generate summary statistics
def generate_summary_statistics(features):
    summary = features.describe(include='all')
    summary_file = 'summary_statistics.csv'
    summary.to_csv(summary_file)
    print(f"Summary statistics saved to {summary_file}")

# Plot feature correlations
def plot_feature_correlations(features):
    plt.figure(figsize=(12, 10))
    correlation_matrix = features.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.show()

# Generate feature report
def generate_feature_report(features_file):
    features = load_features(features_file)
    plot_feature_distributions(features)
    generate_summary_statistics(features)
    plot_feature_correlations(features)

# Example usage
if __name__ == "__main__":
    features_file = r'D:\Earthquak_project\features(1).csv'  # Path to your features CSV
    generate_feature_report(features_file)
