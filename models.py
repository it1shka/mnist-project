# All my models are stored here

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

class ConvolutionNetwork(nn.Module):
  '''Network with convolutional layers'''
  def __init__(self):
    super().__init__()
    self.convolution_stack = nn.Sequential(
      # convolutional part
      nn.Conv2d(1, 10, kernel_size=3),
      nn.ReLU(),
      nn.Conv2d(10, 20, kernel_size=3),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2)
    )
    self.dense_stack = nn.Sequential(
      # dense part
      nn.Linear(2880, 128),
      nn.ReLU(),
      nn.Dropout(0.1),
      nn.Linear(128, 10),
      nn.Softmax(dim=1)
    )
    self.flatten = nn.Flatten()
  
  def forward(self, x):
    x = self.convolution_stack(x)
    x = self.flatten(x)
    x = self.dense_stack(x)
    return x

class CNN(nn.Module):
  '''Network created with some help'''
  def __init__(self):
    super().__init__()
    self.convolution_stack = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),
      nn.Dropout2d(0.25)
    )
    self.flatten = nn.Flatten()
    self.dense_stack = nn.Sequential (
      nn.Linear(64 * 14 * 14, 128),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(128, 10),
      nn.Softmax(dim=1)
    )

  def forward(self, x):
    x = self.convolution_stack(x)
    x = self.flatten(x)
    x = self.dense_stack(x)
    return x

class DigitRecognitionCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.extraction_base = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.flatten = nn.Flatten()
    self.classification_head = nn.Sequential(
      nn.Linear(64 * 7 * 7, 128),
      nn.ReLU(),
      nn.Dropout(0.1),
      nn.Linear(128, 10),
    )
    self.softmax = nn.Softmax(dim=1)
  
  def forward(self, x):
    x = self.extraction_base(x)
    x = self.flatten(x)
    x = self.classification_head(x)
    return self.softmax(x)
  