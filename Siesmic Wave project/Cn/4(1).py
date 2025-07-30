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

# Load segments from CSV in chunks
def load_and_process_csv_chunks(csv_file, chunk_size=1000):
    features_list = []
    sample_rate = 1.0  # Adjust based on your data

    # Process the CSV file in chunks
    for chunk_index, chunk in enumerate(pd.read_csv(csv_file, header=None, chunksize=chunk_size)):
        print(f"Processing chunk {chunk_index + 1}")

        for i, row in chunk.iterrows():
            row = row.values
            num_values = len(row)
            
            # Determine segment dimensions based on the number of values
            side_length = int(np.sqrt(num_values))
            if side_length**2 != num_values:
                print(f"Warning: Row {i} length {num_values} is not a perfect square. Skipping this row.")
                continue

            # Convert row to numeric values
            try:
                row = np.array(row, dtype=np.float64)
            except ValueError:
                print(f"Warning: Row {i} contains non-numeric values. Skipping this row.")
                continue
            
            # Reshape the flattened array into 2D segments
            segment = row.reshape((side_length, side_length))
            features = extract_features_from_segment(segment, sample_rate)
            features['segment'] = i + 1  # or use a unique identifier if needed
            features_list.append(features)
    
    return features_list

# Save features to CSV
def save_features_to_csv(features_list, output_file):
    df = pd.DataFrame(features_list)
    df.to_csv(output_file, index=False)

# Main function
if __name__ == "__main__":
    csv_file = r'D:\Earthquak_project\preprocessed_segments.csv (1)'  # Path to your CSV with preprocessed segments
    output_file = r'D:\Earthquak_project\features(1).csv'  # Output file for features
    
    # Load and process the CSV file in chunks
    features_list = load_and_process_csv_chunks(csv_file)
    
    # Save the extracted features to CSV
    save_features_to_csv(features_list, output_file)
