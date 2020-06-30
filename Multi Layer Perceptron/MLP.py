import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import helper as ut

class NeuralNet:
  bias = {}
  weights = {}
  X_train, X_test = [], []
  y_train, y_test = [], []
  hidden_layer_sizes = []
  layer_sizes = []
  A, df = {},{}
  activation = ''
  learning_rate = 0.1
  epoch = 3000

  def __init__(self, X_train = None, y_train = None, X_test = None, y_test = None, 
                hidden_layer_sizes = [4,3,2] , activation='identity', learning_rate=0.01, epoch=2000):
    self.X_train = X_train
    self.y_train = y_train
    self.X_test = X_test
    self.y_test = y_test
    self.hidden_layer_sizes = hidden_layer_sizes
    self.activation = activation
    self.learning_rate = learning_rate
    self.epoch = epoch
    self.layer_sizes =  [len ( X_train )] + hidden_layer_sizes + [len( y_train )] 
    self._weights_initialization()
    self.train_errors = []
    self.test_errors = []
    self.dW = {}
    self.db = {}
    self.delta = {}

  def _weights_initialization(self):
    for i in range (1,len(self.layer_sizes) ):
      self.weights["W"+str(i)] = np.random.normal(loc=0.0,scale=1.0 / np.sqrt(self.layer_sizes[i]),
                                            size=(self.layer_sizes[i], self.layer_sizes[i-1]))
      self.bias["b"+str(i)] = np.zeros( (self.layer_sizes[i],1) )

  def feed_forward(self, X ,y):
    '''
    Implementation of the Feedforward
    '''
    Z = {}
    input_layer = X
    for i in range(1,len(self.layer_sizes)):
      Z["Z"+str(i)] = np.dot(self.weights["W"+str(i)],input_layer) + self.bias["b"+str(i)]
      if( i == len(self.hidden_layer_sizes) ):
        self.A["A"+str(i)],self.df["df"+str(i)] = ut.sigmoid(Z["Z"+str(i)])
        
        
      else:
        self.A["A"+str(i)],self.df["df"+str(i)] = ut.tanh(Z["Z"+str(i)])
      input_layer = self.A["A"+str(i)]
    error = ut.entropy_loss(self.A["A"+str(len(self.hidden_layer_sizes)+1)],y)
    return  error, self.A["A"+str(len(self.hidden_layer_sizes)+1)]

  def back_propagation(self, X, y):
    L = len(self.hidden_layer_sizes)+1
    m = y.shape[1]
    self.delta["D"+str(L)] = np.multiply((self.A["A"+str(L)] - y),self.df["df"+str(L)])
    self.dW["dW"+str(L)] = 1/m * np.dot(self.delta["D"+str(L)],self.A["A"+str(L-1)].T)
    self.db["db"+str(L)] = 1/m * np.sum(self.delta["D"+str(L)],axis=1,keepdims=True)
    for l in range(L-1,0,-1):
      self.delta["D"+str(l)] = np.multiply(np.dot(self.weights["W"+str(l+1)].T,self.delta["D"+str(l+1)]),self.df["df"+str(l)])

      if (l==1):
        self.dW["dW"+str(l)]=np.dot(self.delta["D"+str(l)],X.T)
      else:
        self.dW["dW"+str(l)]=np.dot(self.delta["D"+str(l)],self.A["A"+str(l-1)].T)
      self.db["db"+str(l)] = 1/m * np.sum(self.delta["D"+str(l)],keepdims=True,axis=1)

    for l in range(1,L+1):
      self.weights["W"+str(l)] = self.weights["W"+str(l)] - self.learning_rate*self.dW["dW"+str(l)]
      self.bias["b"+str(l)] = self.bias["b"+str(l)] - self.learning_rate*self.db["db"+str(l)]
        
  def train (self, X_train,y_train,X_test,y_test):
    mean_error_train, mean_error_test  = [], []
    for i in range(self.epoch):
      error_train, error_test = [], []
      error , A = self.feed_forward(X_train, y_train)
      self.train_errors.append(error)
      self.back_propagation(X_train,y_train)
  def data_output(self):
    probas = self.A["A"+str(len(self.hidden_layer_sizes)+1)].T
    for i in range(0,probas.shape[0]):
      for j in range(0,probas.shape[1]):
          maxElement = np.amax(probas[i])
          if probas[i][j]==maxElement:
            probas[i][j]=1
          else:
            probas[i][j]=0
    return probas
  
  def efficiency(self):
    prob = self.data_output()
    counter = 0
    tr_test = self.y_train.T
    for i in range(0,prob.shape[0]):
        if (prob[i][0] == tr_test[i][0]) and (prob[i][1] == tr_test[i][1]) and (prob[i][2] == tr_test[i][2]):
            counter+=1
            
    percent = counter/prob.shape[0]
    print("Efficiency: {} %".format(percent*100))

