import tkinter as tk
from model import Nexus
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

nb_layers = 5
nb_neurons = 5
memory_size = 5

# Create model, optimizer, and initialize signals for all blocks
def reset_model():
    global model, optimizer, signal
    model = Nexus(nb_layers, nb_neurons, memory_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    signal = model.current_state().detach().numpy()
    update_plot()

def get_signals_array():
    return model.current_state().detach().numpy()

def get_mindstate_array():
    # MindState is of shape (nb_layers, nb_neurons)
    return model.MindState.detach().numpy()

def get_pastself_array():
    # PastSelf is of shape (memory_size, nb_neurons, nb_layers)
    # For visualization, stack along the vertical axis:
    past = model.PastSelf.detach().numpy()  # shape: (memory_size, nb_neurons, nb_layers)
    past_reshaped = past.reshape(memory_size * nb_neurons, nb_layers)
    return past_reshaped

def update_plot():
    # Update PastSelf heatmap (no annotations)
    past_data = get_pastself_array()
    heatmap_past.set_array(past_data)
    
    # Update MindState heatmap and its annotations
    mind_data = get_mindstate_array()
    heatmap_mind.set_array(mind_data)
    for txt in ax_mind.texts:
        txt.remove()
    for i in range(nb_layers):
        for j in range(nb_neurons):
            val = mind_data[i, j]
            ax_mind.text(j, i, f"{val:.2f}", ha='center', va='center',
                         color='white' if val < 0.5 else 'black')
    
    # Update states heatmap and its annotations
    data = get_signals_array()
    heatmap_states.set_array(data)
    for txt in ax_states.texts:
        txt.remove()
    for i in range(nb_neurons):
        for j in range(nb_layers):
            val = data[i, j]
            ax_states.text(j, i, f"{val:.2f}", ha='center', va='center', 
                           color='white' if val < 0.5 else 'black')
    
    canvas.draw()

def step_forward():
    global signal
    signal = model.step().detach().numpy()
    update_plot()

# Tkinter UI Setup
root = tk.Tk()
root.title("Nexus Network")

frame = tk.Frame(root)
frame.pack()

# Create a figure with three subplots arranged in one row:
# PastSelf (left), MindState (middle), and states (right)
fig, (ax_past, ax_mind, ax_states) = plt.subplots(1, 3, figsize=(12, 4))

# PastSelf heatmap setup (left)
past_data = np.zeros((memory_size * nb_neurons, nb_layers))
heatmap_past = ax_past.imshow(past_data, cmap='viridis', vmin=-1, vmax=1)
ax_past.set_title("PastSelf (stacked)")
ax_past.set_xticks(range(nb_layers))
# For y-axis, label memory slices if desired:
y_ticks = [i * nb_neurons + nb_neurons // 2 for i in range(memory_size)]
ax_past.set_yticks(y_ticks)
ax_past.set_yticklabels([f"Memory {i+1}" for i in range(memory_size)])
# Remove colorbar for PastSelf

# MindState heatmap setup (middle)
mind_data = np.zeros((nb_layers, nb_neurons))
heatmap_mind = ax_mind.imshow(mind_data, cmap='viridis', vmin=-1, vmax=1)
ax_mind.set_title("MindState")
# Remove colorbar for MindState

# States heatmap setup (right)
state_data = np.zeros((nb_neurons, nb_layers))
heatmap_states = ax_states.imshow(state_data, cmap='viridis', vmin=-1, vmax=1)
ax_states.set_title("Current States")
ax_states.set_xticks(range(nb_layers))
ax_states.set_yticks(range(nb_neurons))
ax_states.set_yticklabels([f"Layer {i+1}" for i in range(nb_neurons)])
plt.colorbar(heatmap_states, ax=ax_states)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Buttons
tk.Button(frame, text="Step Forward", command=step_forward).pack(side=tk.LEFT)
tk.Button(frame, text="Reset", command=reset_model).pack(side=tk.LEFT)
tk.Button(frame, text="Quit", command=root.destroy).pack(side=tk.LEFT)

# Initialize model after UI setup
reset_model()

root.mainloop()
