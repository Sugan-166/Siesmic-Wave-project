#image to CSV format

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import os

# Load image and convert to grayscale
def load_image(file_path):
    with Image.open(file_path) as img:
        img = img.convert('L')  # Convert to grayscale
        data = np.array(img)  # Convert image to numpy array
    return data

# Apply Butterworth filter for noise reduction
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Normalize data
def normalize_data(data):
    scaler = MinMaxScaler()
    data_reshaped = data.reshape(-1, 1)  # Reshape for scaler
    normalized_data = scaler.fit_transform(data_reshaped)
    return normalized_data.reshape(data.shape)  # Reshape back to original shape

# Segment data
def segment_data(data, segment_height, segment_width):
    num_segments_y = data.shape[0] // segment_height
    num_segments_x = data.shape[1] // segment_width
    segments = []
    
    for i in range(num_segments_y):
        for j in range(num_segments_x):
            segment = data[i*segment_height:(i+1)*segment_height, j*segment_width:(j+1)*segment_width]
            segments.append(segment)
    
    return segments

# Save all segments into a single CSV file
def save_segments_as_single_csv(segments, output_file):
    combined_data = []
    for i, segment in enumerate(segments):
        flattened_segment = segment.flatten()
        combined_data.append(pd.Series(flattened_segment, name=f'segment_{i+1}'))
    
    # Concatenate all series into a single DataFrame
    df = pd.concat(combined_data, axis=1)
    df.to_csv(output_file, index=False)

# Process all images in a folder
def preprocess_images_in_folder(folder_path, output_file, cutoff_freq=0.1, sample_rate=1.0, segment_height=64, segment_width=64):
    segments = []
    
    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # Process only image files
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Processing {file_path}...")
            data = load_image(file_path)
            
            # Noise reduction (apply filter on each row or column as appropriate)
            filtered_data = np.apply_along_axis(butter_lowpass_filter, 1, data, cutoff_freq, sample_rate)
            filtered_data = np.apply_along_axis(butter_lowpass_filter, 0, filtered_data, cutoff_freq, sample_rate)
            
            # Normalize data
            normalized_data = normalize_data(filtered_data)
            
            # Segment data
            image_segments = segment_data(normalized_data, segment_height, segment_width)
            segments.extend(image_segments)
    
    # Save all segments into a single CSV file
    save_segments_as_single_csv(segments, output_file)

# Example usage
if __name__ == "__main__":
    folder_path = r"D:\Earthquak_project\seismic_signals"  # Replace with your folder path
    output_file = 'preprocessed_segments.csv (1)'  # Output file for all segments
    preprocess_images_in_folder(folder_path, output_file)
