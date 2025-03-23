import torch
import torch.nn as nn
import torch.optim as optim

# Define circular model with blocks (each block = Linear + ReLU)
class Nexus(nn.Module):
    def __init__(self, nb_layers, nb_neurons, memory_size):
        super().__init__()
        self.states = (2 * torch.rand(nb_neurons, nb_layers)) - 1
        self.MindState = (2 * torch.rand(nb_layers, nb_neurons)) - 1
        self.PastSelf = torch.stack([self.states.clone() for _ in range(memory_size)])
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(nb_neurons, nb_neurons),
                nn.LeakyReLU()
            ) for _ in range(nb_layers)
        ])
        
        # Convolution to update MindState:
        # We'll use a Conv1d layer with kernel_size = memory_size (which will collapse that dimension)
        self.mind_conv = nn.Conv1d(
            in_channels=nb_layers,      # Treating nb_layers as channels
            out_channels=nb_layers,     # Keeping the same number of channels
            kernel_size=memory_size     # Kernel size equals memory_size to collapse it
        )
    
    def current_state(self):
        return self.states
        
    def update_mindstate(self):
        # Reshape PastSelf:
        # Original PastSelf shape: (memory_size, nb_neurons, nb_layers)
        # Permute to: (nb_neurons, nb_layers, memory_size)
        past_input = self.PastSelf.permute(1, 2, 0)
        
        # Apply the convolution. Expected output shape: (nb_neurons, nb_layers, 1)
        conv_output = self.mind_conv(past_input)
        
        # Apply activation (optional, here using Tanh)
        conv_output = torch.tanh(conv_output)
        
        # Remove the last dimension (sequence length) to get shape: (nb_neurons, nb_layers)
        conv_output = conv_output.squeeze(2)
        
        # Transpose to obtain MindState shape: (nb_layers, nb_neurons)
        self.MindState = conv_output.transpose(0, 1)
    
    def step(self):
        # Process each layer: new_states will have shape (nb_layers, nb_neurons)
        new_states = torch.stack([self.layers[idx](self.states[idx - 1]) for idx in range(len(self.layers))])
        self.states = torch.mm(new_states, self.MindState)
        self.states = torch.tanh(self.states)
        
        # Update PastSelf: remove the oldest state and append the new one
        self.PastSelf = torch.cat([self.PastSelf[1:], self.states.unsqueeze(0)], dim=0)
        
        # Dynamically update MindState using the convolutional network
        self.update_mindstate()
        
        print("\n----- State -----\n\n", self.states)
        print("\n----- MindState -----\n\n", self.MindState)
        print("\n----- PastSelf -----\n\n", self.PastSelf)
        
        return self.states
