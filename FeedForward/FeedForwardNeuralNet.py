import torch
import torch.nn as nn

class Projection_Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Projection_Network, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Linear function  # LINEAR
        x = torch.from_numpy(x)
        out = self.fc1(x)
        return out.detach().numpy()