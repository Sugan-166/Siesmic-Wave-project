
# Signal Simulation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load seismic data from CSV file and ensure all values are numeric
def load_seismic_data(csv_file):
    data = pd.read_csv(csv_file)
    
    # Convert to numeric, ignoring non-numeric values like headers or invalid data
    data = data.apply(pd.to_numeric, errors='coerce')
    
    # Drop any rows or columns that are completely NaN (optional)
    data = data.dropna(axis=0, how='all')  # Drop rows with all NaNs
    data = data.dropna(axis=1, how='all')  # Drop columns with all NaNs
    
    return data.values

# Function to update the plot for each frame in the animation
def update_wave(frame, line, seismic_signals, t):
    line.set_ydata(seismic_signals[frame])  # Update y-data for the current frame
    return line,

# Main simulation function
def simulate_seismic_wave(csv_file, output_file=None):
    seismic_signals = load_seismic_data(csv_file)  # Load signals
    num_frames, num_samples = seismic_signals.shape
    
    # Adjust time axis according to number of samples
    t = np.linspace(0, num_samples / 100, num_samples)  # Assuming 100 Hz sampling rate
    
    # Set up the figure and plot
    fig, ax = plt.subplots()
    ax.set_xlim(0, t[-1])  # Time axis limit
    ax.set_ylim(np.nanmin(seismic_signals), np.nanmax(seismic_signals))  # Amplitude axis limit
    
    line, = ax.plot(t, seismic_signals[0], lw=2)  # Initial plot with the first signal
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Seismic Wave Simulation')

    # Create animation
    ani = FuncAnimation(fig, update_wave, frames=range(num_frames),
                        fargs=(line, seismic_signals, t), interval=100, blit=True)

    # Display the animation
    plt.show()

    # Optionally save the animation as a file
    if output_file:
        try:
            ani.save(output_file, writer='imagemagick')  # Ensure ImageMagick is installed
            print(f'Animation saved to {output_file}')
        except Exception as e:
            print(f'Error saving animation: {e}')

# Example usage
csv_file = r'D:\Earthquak_project\preprocessed_segments.csv'  # Replace with your CSV file path
output_file = r'D:\Earthquak_project\seismic_wave_simulation.gif'  # Replace with your desired output file path
simulate_seismic_wave(csv_file, output_file)
