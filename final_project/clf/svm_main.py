import numpy as np
from svm import SVM
from read_data import Data
from plotBoundary import plotDecisionBoundary 
import matplotlib.pyplot as plt
from gen_data import Data_gen

clf_sel = 'svm_rbf'

plot = False
plot_dec_bound = False # prints dec bound for linear svm, requires t_steps = 1
test_model = True
generate_data = False # generates sinusodial train and test data and predicts traj

dim = 2
t_steps = 10 # trajectory sniplet length


      
train_accs = []
val_accs = []
test_accs = []

## SVM with RBF kernel
for t_steps in [25]:
# for t_steps in [2,10,25,50]:
  # Read training data
  x = -1 * np.ones((0, dim * t_steps))
  y = -1 * np.ones((0,))

  for i in range(1,2):
    train_file = 'clusters_'+str(i)
    data = Data(train_file, verbose=False, _t_steps=t_steps)
    x_i,y_i = data.get_XY()
    
    # y element of {0,1}
    x = np.append(x, x_i, axis=0)
    y = np.append(y, y_i, axis=0)

  # Get val and test data
  val_file = 'clusters_2'
  val_data = Data(val_file, verbose=False, _t_steps=t_steps)
  x_val,y_val = val_data.get_XY()

  test_file = 'clusters_3'
  test_data = Data(test_file, verbose=False, _t_steps=t_steps)
  x_test,y_test = test_data.get_XY()
  for C in [10]:
  # for C in np.logspace(-3,3,7):
    gamma = 'auto'
    kernel = 'rbf'
    clf = SVM(C, gamma, kernel)
    clf.train(x,y)

    train_acc = clf.score(x, y)
    train_accs.append(train_acc)
    val_acc = clf.score(x_val, y_val)
    val_accs.append(val_acc)
    test_acc = clf.score(x_test, y_test)
    test_accs.append(test_acc)
    print "t_steps: %i, C: %f, Train: %.3f, Val: %.3f, Test: %.3f" %(t_steps, C, train_acc,val_acc,test_acc)



