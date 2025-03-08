import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
import threading
import time
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define circular model
class CircularNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU()
        ])

    def forward(self, x):
        activations = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):  # Check if the layer is ReLU
                activations.append(x.detach().numpy())  # Store activations for ReLU layers
        return x, activations

# Create model and optimizer
model = CircularNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
running = False

# 3D Plot Setup
def setup_plot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 9])
    return fig, ax

fig, ax = setup_plot()
scat = ax.scatter([], [], [])

def update_plot(activations):
    def update():
        ax.clear()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, 9])

        x, y, z = [], [], []
        layer_points = []
        for layer_idx, layer_acts in enumerate(activations):
            layer_x, layer_y, layer_z = [], [], []
            for neuron_idx, act in enumerate(layer_acts):
                theta = (layer_idx / len(activations)) * (2 * np.pi)
                radius = np.abs(act)  # Distance from center
                x_pos = radius * np.cos(theta)
                y_pos = radius * np.sin(theta)
                x.append(x_pos)
                y.append(y_pos)
                z.append(neuron_idx)
                layer_x.append(x_pos)
                layer_y.append(y_pos)
                layer_z.append(neuron_idx)
            layer_points.append((layer_x, layer_y, layer_z))

        # Plot neurons
        ax.scatter(x, y, z, c='b', marker='o')

        # Draw edges between neurons in each layer
        # Draw edges between neurons of the same index in neighboring layers
        for layer_idx in range(len(layer_points) - 1):  # Loop through layers, stopping before the last
            for neuron_idx in range(len(layer_points[layer_idx][0])):  # Neurons in each layer
                x0, y0, z0 = layer_points[layer_idx][0][neuron_idx], layer_points[layer_idx][1][neuron_idx], layer_points[layer_idx][2][neuron_idx]
                x1, y1, z1 = layer_points[layer_idx + 1][0][neuron_idx], layer_points[layer_idx + 1][1][neuron_idx], layer_points[layer_idx + 1][2][neuron_idx]
                
                # Draw edge between neuron from layer_idx and the same neuron_idx in the next layer
                ax.plot([x0, x1], [y0, y1], [z0, z1], 'r')

        plt.draw()

    root.after(0, update)  # Schedule update in the main Tkinter thread


def run_model():
    global running
    input_data = torch.randn(10)  # Random input
    while running:
        optimizer.zero_grad()
        output, activations = model(input_data)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        
        print("Neuron Activations:")
        for i, act in enumerate(activations):
            print(f"Layer {i+1}: {act}")
        
        update_plot(activations)
        time.sleep(1)  # Delay to prevent excessive updates

def start():
    global running
    if not running:
        running = True
        threading.Thread(target=run_model, daemon=True).start()

def stop():
    global running
    running = False

# Tkinter UI
root = tk.Tk()
root.title("Circular NN Controller")

tk.Button(root, text="Start", command=start).pack()
tk.Button(root, text="Stop", command=stop).pack()

canvas = tk.Canvas(root, width=500, height=500)
canvas.pack()

def animate(i):
    plt.pause(0.01)

ani = FuncAnimation(fig, animate, interval=1000)
tk.Button(root, text="Show 3D Plot", command=lambda: plt.show()).pack()

root.mainloop()