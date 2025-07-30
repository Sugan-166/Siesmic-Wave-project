#featureextraction
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis

# Extract time-domain features
def extract_time_domain_features(segment):
    features = {}
    features['mean'] = np.mean(segment)
    features['std'] = np.std(segment)
    features['variance'] = np.var(segment)
    features['max'] = np.max(segment)
    features['min'] = np.min(segment)
    features['skewness'] = skew(segment.flatten())
    features['kurtosis'] = kurtosis(segment.flatten())
    return features

# Extract frequency-domain features
def extract_frequency_domain_features(segment, sample_rate):
    num_samples = segment.size
    yf = fft(segment.flatten())
    xf = fftfreq(num_samples, 1 / sample_rate)
    
    # Magnitude spectrum
    magnitude_spectrum = np.abs(yf)
    
    features = {}
    features['mean_freq'] = np.sum(xf * magnitude_spectrum) / np.sum(magnitude_spectrum)
    features['max_freq'] = np.max(xf)
    features['mean_magnitude'] = np.mean(magnitude_spectrum)
    features['max_magnitude'] = np.max(magnitude_spectrum)
    return features

# Extract all features from a segment
def extract_features_from_segment(segment, sample_rate):
    time_domain_features = extract_time_domain_features(segment)
    frequency_domain_features = extract_frequency_domain_features(segment, sample_rate)
    
    all_features = {**time_domain_features, **frequency_domain_features}
    return all_features

# Save features to CSV
def save_features_to_csv(features_list, output_file):
    df = pd.DataFrame(features_list)
    df.to_csv(output_file, index=False)

# Example usage
if __name__ == "__main__":
    # Replace with your actual segment data and sample rate
    sample_rate = 1.0  # Adjust based on your data
    # Example segment data (replace with actual data)
    segments = [np.random.rand(64, 64) for _ in range(10)]  # Replace with actual preprocessed image segments
    
    features_list = []
    for i, segment in enumerate(segments):
        features = extract_features_from_segment(segment, sample_rate)
        features['segment'] = i + 1
        features_list.append(features)
    
    output_file = r'D:\Earthquak_project\features.csv'  # Output file for features
    save_features_to_csv(features_list, output_file)
