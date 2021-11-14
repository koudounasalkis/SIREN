import torch
import torch.nn as nn
import numpy as np
from loss_functions import VGGPerceptualLoss

class Sine_layer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, omega_0=30., is_first=False):
        super().__init__()
        
        self.in_features = in_features
        self.omega_0 = omega_0
        self.is_first = is_first

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)
    
    def forward(self, x):
        y = torch.sin(self.omega_0 * self.linear(x))
        return y
    
    def forward_with_intermediate(self, input): 
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

class Siren(nn.Module):
    def __init__(self, in_features=2, out_features=1, n_hidden_layers=2, hidden_features=256, first_omega_0=30., hidden_omega_0=30., outermost_linear=True, with_vgg_loss=False, with_activations=False):
        '''Our classic SIREN network for this project'''
        super().__init__()
        
        if with_activations:
            hidden_features = 2048
        
        self.in_features = in_features
        self.out_features = out_features
           
        if with_vgg_loss:
            self.vgg_loss = VGGPerceptualLoss() # when the net is sent to CUDA, vgg loss is too
        else:
            self.vgg_loss = None
            
        self.net = []

        # First layer
        self.net.append(Sine_layer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        # Hidden layers
        for i in range(n_hidden_layers):
            self.net.append(Sine_layer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        # Last layer
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(Sine_layer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        if self.vgg_loss is not None:
            sidelength = x['sidelength']
            gt = x['gt']
            coords = x['coords']

            coords = coords.clone().detach().requires_grad_(True)
            y = self.net(coords)

            # Adapt for VGG
            _y = y.view(sidelength, sidelength).unsqueeze(0).unsqueeze(0)
            _gt = gt.view(sidelength, sidelength).unsqueeze(0).unsqueeze(0)

            loss = self.vgg_loss(_y, _gt)
            return y, coords, loss
        else:
            x = x.clone().detach().requires_grad_(True)
            y = self.net(x)
            return y, x
        
    def forward_with_activations(self, coords):
        activations = {}

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, Sine_layer):
                x, intermed = layer.forward_with_intermediate(x)
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
