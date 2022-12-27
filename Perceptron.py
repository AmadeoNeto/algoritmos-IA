import numpy as np
import torch

class Perceptron:

  def __init__(self,activation_function=torch.sign):
    self.weights = None
    self.activation_function = activation_function

  def train(self,x_train:list, y_train:list,
            learning_rate:float=0.001,
            max_epochs:int=500):
    # Convert the training sample to tensors and add w0=-1 to x
    x = torch.tensor([[-1] + xi for xi in x_train],dtype=torch.float)
    y = torch.tensor(y_train)

    # Learning algorithm
    self.weights = torch.rand(x.shape[1])
    epoch = 0
    error_found = True

    while error_found and epoch < max_epochs:
      print(f'\nStarting {epoch}th epoch:')
      error_found = False

      for xi,yi in zip(x,y):
        yi = yi.item()
        print(self.weights, xi)
        tension = torch.dot(self.weights,xi)
        pred = self.activation_function(tension)

        print('pred',pred,'| yi',yi,'| tension:',tension)

        if pred != yi:
          self.weights = self.weights + learning_rate * (yi - pred)*xi
          error_found = True
          print('new weights:',self.weights)
        
      epoch = epoch + 1
  

  def predict(self,x):
    if type(x) is list:
      x = torch.tensor([-1] + x,dtype=torch.float)

    tension = torch.dot(self.weights,x)
    pred = self.activation_function(tension)
    return pred
