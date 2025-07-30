#image generation

import numpy as np
import matplotlib.pyplot as plt
import os

# Define parameters
fs = 100.0  # Sampling frequency (Hz)
t = np.arange(0, 10, 1/fs)  # Time array (10 seconds)
num_signals = 100  # Number of signals to generate

# Create output directory if it doesn't exist
output_dir = 'seismic_signals'
os.makedirs(output_dir, exist_ok=True)

# Function to generate and plot a seismic signal
def generate_and_plot_signal(index):
    # Random parameters for each signal
    freq = np.random.uniform(2.0, 10.0)  # Signal frequency (Hz)
    amp = np.random.uniform(0.5, 2.0)    # Signal amplitude
    noise_amp = np.random.uniform(0.1, 0.5)  # Noise amplitude
    
    # Generate P-wave
    p_wave = amp * np.sin(2 * np.pi * freq * t)
    
    # Generate S-wave with a frequency that is 1.5 times the P-wave frequency
    s_wave = amp * np.sin(2 * np.pi * (freq * 1.5) * t)
    
    # Add noise
    noise = noise_amp * np.random.randn(len(t))
    
    # Combine P-wave, S-wave, and noise
    seismic_signal = p_wave + s_wave + noise
    
    # Plot the signal
    plt.figure(figsize=(12, 6))
    plt.plot(t, seismic_signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Synthetic Seismic Signal {index + 1}')
    plt.tight_layout()
    
    # Save the plot as an image file
    file_path = os.path.join(output_dir, f'seismic_signal_{index + 1}.png')
    plt.savefig(file_path)
    plt.close()  # Close the figure to avoid memory issues

# Generate and save all seismic signal images
for i in range(num_signals):
    generate_and_plot_signal(i)

print(f'Generated and saved {num_signals} seismic signal images to {output_dir}')
