import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def softmax(z):
    shiftz = z - np.max(z)
    exps = np.exp(shiftz)
    return exps / np.sum(exps)

def softmax_gradient(z):
    """Computes the gradient of the softmax function."""
    Sz = softmax(z)
    D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
    return D

def relu(Z):
  A = np.maximum(0,Z)
  assert(A.shape == Z.shape)
  df = np.array(Z,copy = True)
  for i in range(df.shape[0]):
    for j in range(df.shape[1]):
      if(df[i][j]>=0):
        df[i][j]=1
      else:
        df[i][j]=0
 
  return A,df

def tanh(Z):
   A = np.empty(Z.shape)
   A = 2.0/(1 + np.exp(-2.0*Z)) - 1 # A = np.tanh(Z)
   df = 1-A**2
   return A,df

def softmax(z):
    shiftz = z - np.max(z)
    exps = np.exp(shiftz)
    return exps / np.sum(exps)

def softmax_gradient(z):
    Sz = softmax(z)
    D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
    return D

def sigmoid(z):
  A = 1/(1+np.exp(-z))
  df = A*(1-A)
  return A,df

def entropy_loss(AL,Y):
  m = Y.shape[1]
  cost = (1/m)*np.sum(-np.multiply(Y,np.log(np.absolute(AL)))-np.multiply(1-Y,np.log(np.absolute(1-AL))))
  cost = np.squeeze(cost)
  return cost

def convert(y_hat):
  y_max = np.max(y_hat)
  y = []
  for i in range(len(y_hat)):
    if y_hat[i] == y_max:
      y.append(1)
    else:
      y.append(0)
  return y

def plt_confusion_matrix( confusion_mtx = [], class_names = ['class-0', 'class-1', 'class-2'] ):
  plt.figure(figsize = (8,8))
  sns.set(font_scale=2) # label size
  ax = sns.heatmap(
      confusion_mtx, annot=True, annot_kws={"size": 30}, # font size
      cbar=False, cmap='Blues', fmt='d', 
      xticklabels=class_names, yticklabels=class_names)
  ax.set(title='', xlabel='Actual', ylabel='Predicted')
  plt.show()

