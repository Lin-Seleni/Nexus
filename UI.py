import tkinter as tk
from model import Nexus
from torch import mm
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec

# Global config
nb_layers = 6
nb_neurons = 10
memory_size = 5

# Track the initial core and how many steps have been taken
initial_core = None
step_count = 0

# -------------------------
# Data access functions
# -------------------------
def get_core_array():
    # core shape: (nb_layers, nb_neurons)
    return model.core.detach().numpy()

def get_stabilizer_array():
    # stabilizer shape: (nb_neurons, nb_layers)
    return model.stabilizer.detach().numpy()

def get_thought_array():
    # thought shape: (nb_neurons, nb_neurons)
    return model.thought.detach().numpy()

def get_refiner_array():
    # refiner shape: (nb_neurons, nb_layers)
    return model.refiner.detach().numpy()

def get_newcore_array():
    """
    The 'new core' is the product thought * refiner
    *before* the transpose (as in model.step).
    If no step has been taken yet, just return the initial core
    transposed to match the shape (nb_neurons, nb_layers).
    """
    if step_count == 0:
        return initial_core.numpy().T  # (nb_layers, nb_neurons) -> (nb_neurons, nb_layers)
    else:
        new_core = mm(model.thought, model.refiner)  # shape: (nb_neurons, nb_layers)
        return new_core.detach().numpy()

# -------------------------
# Plot update function
# -------------------------
def update_plot():
    # Retrieve the data arrays
    core_data = get_core_array()          # (nb_layers, nb_neurons)
    stabilizer_data = get_stabilizer_array()  # (nb_neurons, nb_layers)
    thought_data = get_thought_array()    # (nb_neurons, nb_neurons)
    refiner_data = get_refiner_array()    # (nb_neurons, nb_layers)
    newcore_data = get_newcore_array()    # (nb_neurons, nb_layers)
    
    # Update each heatmap
    heatmap_core.set_data(core_data)
    heatmap_refiner.set_data(refiner_data)
    heatmap_stabilizer.set_data(stabilizer_data)
    heatmap_thought.set_data(thought_data)
    heatmap_newcore.set_data(newcore_data)
    
    # Clear old annotations and add new ones
    for ax, data, fmt in [
        (ax_core,       core_data,       "{:.2f}"),
        (ax_refiner,    refiner_data,    "{:.2f}"),
        (ax_stabilizer, stabilizer_data, "{:.2f}"),
        (ax_thought,    thought_data,    "{:.2f}"),
        (ax_newcore,    newcore_data,    "{:.2f}")
    ]:
        # remove any existing text objects
        for txt in ax.texts:
            txt.remove()
        # create new text annotations
        rows, cols = data.shape
        for i in range(rows):
            for j in range(cols):
                val = data[i, j]
                ax.text(j, i, fmt.format(val),
                        ha='center', va='center',
                        fontsize=8,
                        color='white' if abs(val) < 0.5 else 'black')
    
    # Use constrained_layout to keep everything lined up
    fig.canvas.draw_idle()

def step_forward():
    global step_count
    step_count += 1
    model.step()  # updates core, thought, etc.
    update_plot()

def reset_model():
    global model, optimizer, initial_core, step_count
    model = Nexus(nb_layers, nb_neurons, memory_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    step_count = 0
    
    # Store initial core so the 'New Core' subplot can show it (transposed) before any step
    initial_core = model.current_state().detach().clone()
    update_plot()

# -------------------------
# Tkinter + Matplotlib Setup
# -------------------------
root = tk.Tk()
root.title("Nexus Network Multiplication View")

frame = tk.Frame(root)
frame.pack()

# Create figure with a 2 x 3 GridSpec
# We'll place subplots like so:
#
#   Row0, Col0: (empty)        Row0, Col1: Core       Row0, Col2: Refiner
#   Row1, Col0: Stabilizer     Row1, Col1: Thought    Row1, Col2: New Core
#
fig = plt.figure(figsize=(10, 6), constrained_layout=True)
gs = GridSpec(2, 3, figure=fig)

ax_core = fig.add_subplot(gs[0, 1])
ax_core.set_title("Core")
heatmap_core = ax_core.imshow(np.zeros((nb_layers, nb_neurons)), cmap='viridis', vmin=-1, vmax=1)
ax_core.set_aspect("equal", adjustable="box")
ax_core.set_anchor('SW')  # Align core at the bottom left of its Axes

ax_refiner = fig.add_subplot(gs[0, 2])
ax_refiner.set_title("Refiner")
heatmap_refiner = ax_refiner.imshow(np.zeros((nb_neurons, nb_layers)), cmap='viridis', vmin=-1, vmax=1)
ax_refiner.set_aspect("equal", adjustable="box")

ax_stabilizer = fig.add_subplot(gs[1, 0])
ax_stabilizer.set_title("Stabilizer")
heatmap_stabilizer = ax_stabilizer.imshow(np.zeros((nb_neurons, nb_layers)), cmap='viridis', vmin=-1, vmax=1)
ax_stabilizer.set_aspect("equal", adjustable="box")

ax_thought = fig.add_subplot(gs[1, 1])
ax_thought.set_title("Thought")
heatmap_thought = ax_thought.imshow(np.zeros((nb_neurons, nb_neurons)), cmap='viridis', vmin=-1, vmax=1)
ax_thought.set_aspect("equal", adjustable="box")

ax_newcore = fig.add_subplot(gs[1, 2])
ax_newcore.set_title("New Core (pre-transpose)")
heatmap_newcore = ax_newcore.imshow(np.zeros((nb_neurons, nb_layers)), cmap='viridis', vmin=-1, vmax=1)
ax_newcore.set_aspect("equal", adjustable="box")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=5)

tk.Button(btn_frame, text="Step Forward", command=step_forward).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="Reset", command=reset_model).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="Quit", command=root.destroy).pack(side=tk.LEFT, padx=5)

# Initialize everything
reset_model()

root.mainloop()
