import torch
import torch.nn as nn
import torch.optim as optim

# Define circular model with blocks (each block = Linear + ReLU)
class Nexus(nn.Module):
    def __init__(self, nb_layers, nb_neurons, memory_size):
        super().__init__()
        self.core = (2 * torch.rand(nb_layers, nb_neurons)) - 1
        self.stabilizer = (2 * torch.rand(nb_neurons, nb_layers)) - 1
        self.refiner = (2 * torch.rand(nb_neurons, nb_layers)) - 1
        self.thought = (2 * torch.rand(nb_neurons, nb_neurons)) - 1
        self.PastSelf = torch.stack([self.core.clone() for _ in range(memory_size)])
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(nb_neurons, nb_neurons),
                nn.LeakyReLU()
            ) for _ in range(nb_layers)
        ])
    
    def current_state(self):
        return self.core
    
    def step(self):
        # Process each layer: new_core will have shape (nb_layers, nb_neurons)
        print("core : \n", self.core)
        print("layers :\n", [self.layers[idx] for idx in range(len(self.layers))])
        new_core = torch.stack([self.layers[idx](self.core[idx - 1]) for idx in range(len(self.layers))])
        self.thought = torch.mm(self.stabilizer, new_core)
        self.core = torch.mm(self.thought, self.refiner).T
        self.core = torch.tanh(self.core)
        
        # Update PastSelf: remove the oldest state and append the new one
        self.PastSelf = torch.cat([self.PastSelf[1:], self.core.unsqueeze(0)], dim=0)
        
        print("\n----- State -----\n\n", self.core)
        print("\n----- stabilizer -----\n\n", self.stabilizer)
        print("\n----- PastSelf -----\n\n", self.PastSelf)
        
        return self.core
