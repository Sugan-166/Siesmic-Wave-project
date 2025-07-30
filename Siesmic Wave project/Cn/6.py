import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

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
    return scaled_features

# Build neural network model
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Use 'sigmoid' for binary classification; 'linear' for regression
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Use 'mean_squared_error' for regression
    return model

# Train the model
def train_model(model, X_train, X_val):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # Dummy target data for training
    y_train = np.zeros(X_train.shape[0])
    y_val = np.zeros(X_val.shape[0])
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
    return history

# Main function
if __name__ == "__main__":
    features_file = r'D:\Earthquak_project\features.csv'  # Path to your features CSV

    # Load and preprocess data
    features = load_data(features_file)
    X = preprocess_data(features)

    # Split data into training and validation sets
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

    # Build and train model
    model = build_model(input_dim=X_train.shape[1])
    history = train_model(model, X_train, X_val)

    # Save the model
    model.save(r'D:\Earthquak_project\model.h5')

    # Print training history
    print("Training History:")
    print("Loss:", history.history['loss'])
    print("Validation Loss:", history.history['val_loss'])
    print("Accuracy:", history.history['accuracy'])
    print("Validation Accuracy:", history.history['val_accuracy'])
