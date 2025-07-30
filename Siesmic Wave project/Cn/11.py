import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import glob

# Define earthquake-prone locations with longitude and latitude
earthquake_prone_regions = [
    {'location': 'Chile', 'longitude': -70.6506, 'latitude': -33.4569},
    {'location': 'Japan', 'longitude': 138.2529, 'latitude': 36.2048},
    {'location': 'Indonesia', 'longitude': 113.9213, 'latitude': -0.7893},
    {'location': 'California, USA', 'longitude': -119.4179, 'latitude': 36.7783},
    {'location': 'Turkey', 'longitude': 35.2433, 'latitude': 38.9637},
    {'location': 'Mexico', 'longitude': -102.5528, 'latitude': 23.6345}
]

# Function to simulate seismic signals
def simulate_seismic_signals(num_signals=6, signal_length=100):
    seismic_signals = []
    for _ in range(num_signals):
        signal = np.sin(np.linspace(0, 4 * np.pi, signal_length)) * np.exp(-np.linspace(0, 2, signal_length))
        seismic_signals.append(signal)
    return seismic_signals

# Function to find the shapefile in a directory
def find_shapefile(folder_path):
    shapefile_path = glob.glob(os.path.join(folder_path, '*.shp'))
    if not shapefile_path:
        raise FileNotFoundError(f"No shapefile found in directory {folder_path}")
    return shapefile_path[0]

# Plot earthquake data on satellite map with animated seismic signals
def plot_earthquake_map_with_animation(earthquake_prone_regions, seismic_signals, shapefile_folder, output_file):
    # Convert the regions to a pandas DataFrame
    df = pd.DataFrame(earthquake_prone_regions)

    # Convert the DataFrame to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']))

    # Set the initial CRS to WGS84 (longitude/latitude)
    gdf.set_crs(epsg=4326, inplace=True)

    # Find the shapefile in the given folder
    shapefile_path = find_shapefile(shapefile_folder)
    world = gpd.read_file(shapefile_path)

    # Convert the world map to Web Mercator projection
    world = world.to_crs(epsg=3857)

    # Convert earthquake-prone regions to Web Mercator projection
    gdf = gdf.to_crs(epsg=3857)

    # Setup the plot with satellite imagery
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the world boundaries
    world.plot(ax=ax, color='lightgray')

    # Plot earthquake-prone regions
    gdf.plot(ax=ax, color='red', markersize=50, label='Earthquake-Prone Regions')

    # Add satellite imagery as background
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=2)

    # Initialize lines for seismic signals
    lines = [ax.plot([], [], color='blue', alpha=0.6, label=f'Seismic Signal {i+1}')[0] for i in range(len(seismic_signals))]

    # Set axis limits
    x_min, x_max = gdf.geometry.x.min() - 1e7, gdf.geometry.x.max() + 1e7
    y_min, y_max = gdf.geometry.y.min() - 1e7, gdf.geometry.y.max() + 1e7
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    plt.title("Earthquake-Prone Regions with Animated Seismic Signals on Satellite Map")

    # Initialize waveforms for each signal
    waveforms = [ax.plot([], [], color='blue', alpha=0.6)[0] for _ in range(len(seismic_signals))]
    
    # Animation update function
    def update(frame):
        for i, (signal, waveform) in enumerate(zip(seismic_signals, waveforms)):
            x = gdf.geometry.x.iloc[i]
            y = gdf.geometry.y.iloc[i]
            x_values = np.linspace(x - 1e7, x + 1e7, len(signal))
            y_values = np.linspace(y - 1e7, y + 1e7, len(signal))
            waveform.set_data(x_values[:frame], y_values[:frame])
        return waveforms

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(seismic_signals[0]), blit=True, repeat=False)

    # Save the animation
    ani.save(output_file, writer=PillowWriter(fps=10))

    print(f"Animation saved as {output_file}")

# Main function
if __name__ == "__main__":
    # Path to the folder containing shapefile and associated files
    shapefile_folder = r'C:\Users\sugan\Desktop\ne_110m_admin_0_countries'

    # Simulate seismic signals (replace with real data if available)
    seismic_signals = simulate_seismic_signals(num_signals=6, signal_length=100)

    # Output file for saving the animation
    output_file = 'earthquake_animation.gif'

    # Plot earthquake-prone locations on the satellite map with animated seismic signals
    plot_earthquake_map_with_animation(earthquake_prone_regions, seismic_signals, shapefile_folder, output_file)
