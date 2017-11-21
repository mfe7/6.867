import numpy as np
from svm import SVM
from read_data import Data


train_file = 'clusters_2016_2_9'
data = Data(train_file)
x,y = data.get_XY()

plot_nsnips = 1000 # plot no more than 1000 trajectory sniplets to reduce plotting time
data.plot_clf(x,y, plot_nsnips)


clf = 'svm_rbf'
if clf == 'svm_rbf':
  C = 1
  gamma = 'auto'
  kernel = 'rbf'
  svm = SVM(C, gamma, kernel)

  svm.train(x,y)

  train_acc = svm.score(x, y)
  print('[STATUS] Train accuracy of {0:.2f}%'.format(train_acc*100))

test = True
if test == True:
  val_file = 'clusters_2016_2_10'
  val_data = Data(val_file)
  x_val,y_val = val_data.get_XY()
  val_acc = svm.score(x_val, y_val)
  print('[STATUS] Validation accuracy of {0:.2f}%'.format(val_acc*100))

  test_file = 'clusters_2016_2_11'
  test_data = Data(test_file)
  x_test,y_test = test_data.get_XY()
  test_acc = svm.score(x_test, y_test)
  print('[STATUS] Test accuracy of {0:.2f}%'.format(test_acc*100))

