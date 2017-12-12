import numpy as np
from plotBoundary import plotDecisionBoundary 
import matplotlib.pyplot as plt


class Data_gen:
  def __init__(self, _t_steps=10, input_dim=2):
    self.fct = None
    self._t_steps = _t_steps
    self.dim = input_dim
    
    self.n_samples = 0
    self._X = None
  # x range [0,2pi]
  def sin(self,x):
    return sin(x)

  def gen_data(self,fct='sin', n_samples=1000):
    #X = np.zeros((self._t_steps, n_samples, self.dim))
    self.n_samples = n_samples
    # x is concatenated with x1x2
    self._X = np.zeros((self.n_samples, self.dim * self._t_steps))

    for i in range(self.n_samples):
      # range x
      range_x = 1+np.random.rand()*(20-1)
      x1 = np.linspace(start=0, stop=range_x, num=self._t_steps, endpoint=False,dtype=float)
      x2 = np.sin(x1)

      # scale on y
      scale_y = 1 + 5*np.random.rand()
      x2 = np.dot(x2,scale_y)

      # Rotate around positive z-axis
      theta = (2*np.pi) * np.random.rand()
      # Translate in bounding rectangle
      t_bound = 20
      t_x1 = -10 + t_bound * np.random.rand()
      t_x2 = -10 + t_bound * np.random.rand()
      trans = np.array([[np.cos(theta), np.sin(theta), 0.0, t_x1],
        [-np.sin(theta), np.cos(theta), 0.0, t_x2], 
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

      x_augm = np.array([x1, x2, np.ones(self._t_steps), np.ones(self._t_steps)])
      x_augm = np.matmul(trans, x_augm)

      self._X[i,::2] = x_augm[0]
      self._X[i,1::2] = x_augm[1]

    return self._X
