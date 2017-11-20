import numpy as np
from svm import SVM
from read_data import Data


data = Data()
x,y = data.get_XY()

plot_nsnips = 1000
data.plot_clf(x,y, plot_nsnips)


clf = 'None'
if clf == 'svm_rbf':
  C = 1
  gamma = 'auto'
  kernel = 'rbf'
  svm = SVM(C, gamma, kernel)

  svm.train(x,y)

  print svm.score(x, y)
