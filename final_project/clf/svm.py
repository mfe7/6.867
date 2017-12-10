# Ideas
# - show that velocity orientation is reflected in classification
import numpy as np
from sklearn.svm import SVC

class SVM:
  def __init__(self, C, gamma, kernel):
    self._svm = None
    self._C = C
    self._gamma = gamma
    self._kernel = kernel
    self._init_svm()

  def _init_svm(self):
    self._svm = SVC(C=self._C, kernel=self._kernel, degree=3, 
      gamma=self._gamma, coef0=0.0, shrinking=True, probability=False, 
      tol=0.001, cache_size=200, class_weight=None, verbose=False, 
      max_iter=-1, decision_function_shape='ovr', random_state=None)
    
  def train(self, x, y):
    print('[STATUS] Train SVM with {} sniplets...'.format(y.shape[0]))
    self._svm.fit(x, y)
    print('[STATUS] Training finished'.format(y.shape[0]))


  def score(self, x, y):
    acc = self._svm.score(x, y)
    return acc

  def predict(self, x):
    pred = self._svm.predict(x)
    return pred

