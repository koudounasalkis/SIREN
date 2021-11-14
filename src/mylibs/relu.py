import torch
from torch import nn

class Relu_layer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, inplace=False, is_first=False):
        super().__init__()
        
        self.in_features = in_features
        self.is_first = is_first
        self.inplace = inplace

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        nn.init.kaiming_normal_(self.linear.weight, a=0.0, nonlinearity='relu', mode='fan_in')

    def forward(self, x):
        x = self.linear(x)
        y = nn.functional.relu(x, inplace=self.inplace)
        return y

class Relu(nn.Module):
    def __init__(self, n_hidden_layers=2, hidden_features=256):
        '''Out classic SIREN network for this project'''
        super().__init__()

        # Constants
        in_features = 2
        out_features = 1

        self.net = []

        # First layer
        self.net.append(Relu_layer(in_features, hidden_features, inplace=True, is_first=True))

        # Hidden layers
        for i in range(n_hidden_layers):
            self.net.append(Relu_layer(hidden_features, hidden_features, inplace=True, is_first=False))

        # Last layer
        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            nn.init.kaiming_normal_(final_linear.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = x.clone().detach().requires_grad_(True)
        y = self.net(x)
        return y, x
