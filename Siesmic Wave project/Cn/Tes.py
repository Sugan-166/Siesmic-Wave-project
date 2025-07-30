import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

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

# Load dataset
def load_data(features_file):
    features = pd.read_csv(features_file)
    print("Features shape:", features.shape)
    return features

# Preprocess data
def preprocess_data(features):
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler

# Build neural network model
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Use 'sigmoid' for binary classification; 'linear' for regression
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Use 'mean_squared_error' for regression
    return model

# Train the model
def train_model(model, X_train, y_train, X_val, y_val):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
    return history

# Load the trained model
def load_trained_model(model_path):
    model = load_model(model_path)
    return model

# Preprocess new data
def preprocess_new_data(features_file, scaler):
    try:
        features = pd.read_csv(features_file)
        if features.empty:
            raise ValueError("The features file is empty.")
        scaled_features = scaler.transform(features)
        return scaled_features
    except pd.errors.EmptyDataError:
        print("Error: The file is empty or not readable.")
        raise
    except ValueError as e:
        print(f"Error: {e}")
        raise

# Detect and alert based on predictions
def detect_and_alert(model, data):
    predictions = model.predict(data)
    # Assuming binary classification: if prediction > 0.5, alert
    alerts = predictions > 0.5
    return alerts

# Main function for feature extraction
def main_feature_extraction():
    sample_rate = 1.0  # Adjust based on your data
    csv_file = r'D:\Earthquak_project\preprocessed_segments.csv'  # Path to your CSV with preprocessed segments
    output_file = r'D:\Earthquak_project\features.csv'  # Output file for features
    
    # Load preprocessed image segments from CSV
    df = pd.read_csv(csv_file, header=None)
    segments = [row.values.reshape((64, 64)) for _, row in df.iterrows() if len(row.values) == 4096]

    # Extract features from each segment
    features_list = []
    for i, segment in enumerate(segments):
        features = extract_features_from_segment(segment, sample_rate)
        features['segment'] = i + 1
        features_list.append(features)
    
    # Save the extracted features to CSV
    save_features_to_csv(features_list, output_file)

# Main function for model training
def main_model_training():
    features_file = r'D:\Earthquak_project\features.csv'  # Path to your features CSV

    # Load and preprocess data
    features = load_data(features_file)
    X = features.drop(columns=['segment'])  # Assuming 'segment' is not a feature
    y = np.zeros(X.shape[0])  # Dummy target data for training
    X, scaler = preprocess_data(X)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train model
    model = build_model(input_dim=X_train.shape[1])
    history = train_model(model, X_train, y_train, X_val, y_val)

    # Save the model
    model.save(r'D:\Earthquak_project\model.h5')

    # Print training history
    print("Training History:")
    print("Loss:", history.history['loss'])
    print("Validation Loss:", history.history['val_loss'])
    print("Accuracy:", history.history['accuracy'])
    print("Validation Accuracy:", history.history['val_accuracy'])

# Main function for detection and alert
def main_detection_alert():
    model_path = r'D:\Earthquak_project\model.h5'  # Path to your saved model
    features_file = r'D:\Earthquak_project\features.csv'  # Path to your new data

    # Load the trained model
    model = load_trained_model(model_path)
    
    # Load and preprocess new data
    try:
        scaler = StandardScaler()  # Use the same scaler that was used during training
        training_features_file = r'D:\Earthquak_project\features.csv'  # Path to the training features CSV
        features = pd.read_csv(training_features_file)
        if features.empty:
            raise ValueError("The training features file is empty.")
        scaler.fit(features.drop(columns=['segment']))  # Fit scaler on training features

        new_data = preprocess_new_data(features_file, scaler)
    
        # Detect and generate alerts
        alerts = detect_and_alert(model, new_data)
    
        # Output alerts
        for i, alert in enumerate(alerts):
            if alert:
                print(f"Alert: Anomaly detected in sample {i + 1}")
            else:
                print(f"Sample {i + 1}: No anomaly detected")

    except Exception as e:
        print(f"An error occurred: {e}")

# Choose which function to run
if __name__ == "__main__":
    # Run the desired function here
    main_feature_extraction()  # Run feature extraction
    # main_model_training()  # Run model training
    # main_detection_alert()  # Run detection and alert
