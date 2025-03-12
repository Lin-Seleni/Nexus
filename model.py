import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

nb_layers = 10
nb_neurons = 10

# Define circular model with blocks (each block = Linear + ReLU)
class CircularNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Create nb_layers blocks, each consisting of a Linear and a ReLU
        self.layers = nn.ModuleList([
            layer for i in range(nb_layers) for layer in (nn.Linear(nb_neurons, nb_neurons), nn.ReLU())
        ])

    def forward_block(self, x, block_index):
        # Each block is two layers: a Linear layer then a ReLU activation.
        linear = self.layers[2 * block_index]
        relu = self.layers[2 * block_index + 1]
        out = relu(linear(x))
        return out

# Create model, optimizer, and initialize signals for all blocks
def reset_model():
    global model, optimizer, signals
    model = CircularNN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # Initialize each block's signal with a random vector.
    signals = [torch.randn(nb_neurons) for _ in range(nb_layers)]

reset_model()

# Tkinter UI Setup
root = tk.Tk()
root.title("Circular NN Controller")

frame = tk.Frame(root)
frame.pack()

# Heatmap Setup
fig, ax = plt.subplots()

def get_signals_array():
    # Convert each torch tensor to numpy array for display
    return np.stack([s.detach().numpy() for s in signals], axis=0)

heatmap = ax.imshow(get_signals_array(), cmap='viridis', vmin=0, vmax=1)
plt.colorbar(heatmap)
ax.set_title("Neuron Activations Heatmap")
ax.set_xticks([])
ax.set_yticks(range(nb_layers))
ax.set_yticklabels([f"Layer {i+1}" for i in range(nb_layers)])

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Update heatmap with current signals
def update_plot():
    data = get_signals_array()
    heatmap.set_array(data)
    canvas.draw()

# On each step, update every block concurrently.
# For block i, use the previous block's signal (with circular wrap-around)
def step_forward():
    global signals
    new_signals = []
    for i in range(nb_layers):
        # For i=0, signals[-1] (last block) serves as input, achieving a circular connection.
        prev_signal = signals[i-1]
        new_signal = model.forward_block(prev_signal, i)
        new_signals.append(new_signal)
    signals = new_signals
    update_plot()

def reset():
    reset_model()
    update_plot()

# Buttons: one click now performs a full block update (both linear and relu).
tk.Button(frame, text="Step Forward", command=step_forward).pack(side=tk.LEFT)
tk.Button(frame, text="Reset", command=reset).pack(side=tk.LEFT)
tk.Button(frame, text="Quit", command=root.destroy).pack(side=tk.LEFT)

update_plot()
root.mainloop()
