from torch import nn

class DenseNetwork(nn.Module):
  '''Network with all dense layers'''
  def __init__(self):
    super().__init__()
    # defining flatten, activation and output functions 
    self.flatten = nn.Flatten()
    self.activation_function = nn.ReLU()
    self.output_function = nn.Softmax(dim=1)
    # defining the layers
    self.hidden_layer_1 = nn.Linear(28 * 28, 128)
    self.hidden_layer_2 = nn.Linear(128, 64)
    self.output_layer = nn.Linear(64, 10)
  
  def forward(self, x):
    x = self.flatten(x)
    x = self.hidden_layer_1(x)
    x = self.activation_function(x)
    x = self.hidden_layer_2(x)
    x = self.activation_function(x)
    x = self.output_layer(x)
    x = self.output_function(x)
    return x