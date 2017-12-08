import numpy as np
from svm import SVM
from read_data import Data
from rnn import RNN
import tensorflow as tf


train_file = 'clusters_2016_2_9'
data = Data(train_file)
x,y = data.get_XY()
# y element of {0,1}

plot = 1
clf_sel = 'rnn'
test_model = True

#for lr in [0.001]:
#  for n_hidden in [128]:
#    for batch_size in [1]:
      #print('[STATUS] LEARNING RATE = {}'.format(lr))
      #print('[STATUS] N hidden units = {}'.format(n_hidden))
      #print('[STATUS] BATCH SIZE = {}'.format(batch_size))
      
## SVM with RBF kernel
if clf_sel == 'svm_rbf':
  C = 1
  gamma = 'auto'
  kernel = 'rbf'
  clf = SVM(C, gamma, kernel)

  clf.train(x,y)

  train_acc = clf.score(x, y)
  print('[STATUS] Train accuracy of {0:.2f}%'.format(train_acc*100))

## RNN
elif clf_sel == 'rnn':

  input_dim = 2 # x and y position of pedestrian
  n_samples = None # set in initialize_graph
  n_hidden = 128 # num hidden layers = num features
  n_classes = 1 # binary: cross / no-cross, not one-hot encoded, {0,1} instead 
  max_epochs=3

  lr = 0.001 # Learning rate
  batch_size = 1
  max_iter = 10000

  tf.reset_default_graph()
  clf = RNN(input_dim, n_hidden, n_classes, lr, batch_size, max_iter, max_epochs)
  clf.initialize_graph()
  clf.train_and_predict(x, y)

if test_model == True:
  val_file = 'clusters_2016_2_10'
  val_data = Data(val_file, verbose=False)
  x_val,y_val = val_data.get_XY()
  val_acc = clf.score(x_val, y_val)
  print('[STATUS] Validation accuracy of {0:.2f}%'.format(val_acc*100))

  test_file = 'clusters_2016_2_11'
  test_data = Data(test_file, verbose=False)
  x_test,y_test = test_data.get_XY()
  test_acc = clf.score(x_test, y_test)
  print('[STATUS] Test accuracy of {0:.2f}%'.format(test_acc*100))


if plot == 1:
  print('[STATUS] Plot train set]')
  plot_nsnips = 1000 # plot no more than 1000 trajectory sniplets to reduce plotting time
  data.plot_clf(x,y, plot_nsnips)

  print('[STATUS] Plot validation set]')
  data.plot_clf(x_val,y_val, plot_nsnips)

  print('[STATUS] Plot test set')
  data.plot_clf(x_test,y_test, plot_nsnips)

