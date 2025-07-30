
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Step 1: Generate grid for the map (latitude and longitude range)
x = np.linspace(-5, 5, 100)  # X-axis (longitude)
y = np.linspace(-5, 5, 100)  # Y-axis (latitude)
x, y = np.meshgrid(x, y)

# Step 2: Create synthetic wave data (replace this with your seismic data)
# For a moving wave effect, you could use a sinusoidal function.
def generate_wave(t):
    return np.sin(np.sqrt(x**2 + y**2) - t)

# Step 3: Initialize the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(-1, 1)

# Plot the initial surface
wave = generate_wave(0)
surface = ax.plot_surface(x, y, wave, cmap='viridis')

# Step 4: Function to update the surface for animation
def update(t):
    ax.clear()
    wave = generate_wave(t)
    ax.plot_surface(x, y, wave, cmap='viridis')
    ax.set_zlim(-1, 1)  # Keep the z-axis limits consistent

# Step 5: Animate the wave movement
ani = FuncAnimation(fig, update, frames=np.linspace(0, 10, 100), interval=100)

plt.show()
