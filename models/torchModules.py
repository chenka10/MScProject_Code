import torch.nn as nn
import torchvision.models as models
import torch

class BasicFc(nn.Module):
  def __init__(self,in_dim, out_dim, fc_hidden_units=[256, 128],activation = nn.ReLU()):
        super(BasicFc, self).__init__()

        self.fc_layers = nn.ModuleList([])

        if len(fc_hidden_units) == 0:
          self.fc_layers.append(nn.Linear(in_dim,out_dim))  
          return

        self.fc_layers.append(nn.Linear(in_dim,fc_hidden_units[0]))
        self.fc_layers.append(nn.BatchNorm1d(fc_hidden_units[0]))
        self.fc_layers.append(activation)        

        for (i,hidden_unit) in enumerate(fc_hidden_units[1:]):
          self.fc_layers.append(nn.Linear(fc_hidden_units[i],hidden_unit))
          self.fc_layers.append(nn.BatchNorm1d(hidden_unit))
          self.fc_layers.append(activation)

        self.fc_layers.append(nn.Linear(fc_hidden_units[-1],out_dim))

  def forward(self, x):
        # Forward pass through fully connected layers
        for layer in self.fc_layers:
            x = layer(x)
        return x
