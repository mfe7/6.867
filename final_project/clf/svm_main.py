import numpy as np
from svm import SVM
from read_data import Data
from plotBoundary import plotDecisionBoundary 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from gen_data import Data_gen

clf_sel = 'svm_rbf'

plot = False
plot_dec_bound = True # prints dec bound for linear svm, requires t_steps = 1
test_model = True
generate_data = False # generates sinusodial train and test data and predicts traj

dim = 2

##########################
# Plot amount of training data used
############################
# data_pts = []
# test_accs = []
# t_steps = 25 # trajectory sniplet length

# test_file = 'clusters_9'
# test_data = Data(test_file, verbose=False, _t_steps=t_steps)
# x_test,y_test = test_data.get_XY()

# for j in [2,3,4,5,8,22,23,24,25,26,27,28,29]:
#   x = -1 * np.ones((0, dim * t_steps))
#   y = -1 * np.ones((0,))


#   if j < 9:
#     dates = range(1,j)
#   else:
#     dates = range(1,8) + range(21,j)
#   for i in dates:
#       train_file = 'clusters_'+str(i)
#       data = Data(train_file, verbose=False, _t_steps=t_steps)
#       x_i,y_i = data.get_XY()
#       x = np.append(x, x_i, axis=0)
#       y = np.append(y, y_i, axis=0)

#   C = 1
#   gamma = 'auto'
#   kernel = 'rbf'
#   clf = SVM(C, gamma, kernel)

#   clf.train(x,y)
#   print 'Done training'

#   test_acc = clf.score(x_test, y_test)
#   data_pts.append(x.shape[0])
#   test_accs.append(test_acc)
#   print x.shape[0], test_acc

# print "data_pts=",data_pts
# print "test_accs=",test_accs

# data_pts= [1675, 4067, 6787, 9700, 9700, 9700, 9704, 13229, 15459, 19073, 22841, 22841, 22841]
# test_accs= [0.79572338489535943, 0.83303002729754327, 0.84167424931756141, 0.84758871701546856, 0.84758871701546856, 0.84758871701546856, 0.84758871701546856, 0.85304822565969063, 0.8575978161965423, 0.86396724294813465, 0.86396724294813465, 0.86396724294813465, 0.86396724294813465]

# plt.plot(data_pts,test_accs,'r--o')
# plt.xlabel('# Trajectory Snippets')
# plt.ylabel('Test Accuracy (\%)')
# plt.show()

###########################
# Plot predictions
###########################
t_steps = 25 # trajectory sniplet length
x = -1 * np.ones((0, dim * t_steps))
y = -1 * np.ones((0,))
for i in range(1,8) + range(21,29):
    train_file = 'clusters_'+str(i)
    data = Data(train_file, verbose=False, _t_steps=t_steps)
    x_i,y_i = data.get_XY()
    
    # y element of {0,1}
    x = np.append(x, x_i, axis=0)
    y = np.append(y, y_i, axis=0)

test_file = 'clusters_9'
test_data = Data(test_file, verbose=False, _t_steps=t_steps)
x_test,y_test = test_data.get_XY()

C = 1
gamma = 'auto'
kernel = 'rbf'
clf = SVM(C, gamma, kernel)

clf.train(x,y)
print 'Done training'

false_pos = 0
false_neg = 0
true_pos = 0
true_neg = 0

for i in range(x_test.shape[0]):
  pred = clf.predict(x_test[i].reshape(1,-1))
  label = y_test[i]
  xs = x_test[i][::2]
  ys = x_test[i][1::2]
  if pred == 0 and label == 0:
    plt.plot(xs, ys, '-', color = 'green')
    true_neg += 1
  elif pred == 1 and label == 0:
    plt.plot(xs, ys, '-', color = 'red') 
    false_pos += 1
  # if pred == 0 and label == 1:
  #   plt.plot(xs, ys, '-', color = 'orange') 
  #   false_neg += 1
  # elif pred == 1 and label == 1:
  #   plt.plot(xs, ys, '-', color = 'cyan') 
  #   true_pos += 1

print "false_pos:",false_pos,false_pos/float(x_test.shape[0])
print "false_neg:",false_neg,false_neg/float(x_test.shape[0])
print "true_pos:",true_pos,true_pos/float(x_test.shape[0])
print "true_neg:",true_neg,true_neg/float(x_test.shape[0])

green_patch = mpatches.Patch(color='green', label='True negatives')
red_patch = mpatches.Patch(color='red', label='False positivies')
# orange_patch = mpatches.Patch(color='orange', label='False negatives')
# cyan_patch = mpatches.Patch(color='cyan', label='True positives')
plt.legend(handles=[green_patch, red_patch],loc='lower left')
# plt.legend(handles=[orange_patch, cyan_patch],loc='lower left')

plt.xlim([-20,20])
plt.ylim([-20,30])

plt.show()


##########################
# Plot decision boundary
#########################
# t_steps = 1 # trajectory sniplet length
# x = -1 * np.ones((0, dim * t_steps))
# y = -1 * np.ones((0,))
# for i in range(1,2):
#     train_file = 'clusters_'+str(i)
#     data = Data(train_file, verbose=False, _t_steps=t_steps)
#     x_i,y_i = data.get_XY()
    
#     # y element of {0,1}
#     x = np.append(x, x_i, axis=0)
#     y = np.append(y, y_i, axis=0)

# # Get val and test data
# val_file = 'clusters_2'
# val_data = Data(val_file, verbose=False, _t_steps=t_steps)
# x_val,y_val = val_data.get_XY()

# test_file = 'clusters_3'
# test_data = Data(test_file, verbose=False, _t_steps=t_steps)
# x_test,y_test = test_data.get_XY()
# C = 0.1
# gamma = 'auto'
# kernel = 'rbf'
# clf = SVM(C, gamma, kernel)

# clf.train(x,y)

# train_acc = clf.score(x, y)
# print('[STATUS] Train accuracy of {0:.2f}%'.format(train_acc*100))

# if plot_dec_bound == True:
#   values = [0,1]
#   x_plot = x[0:1000,:]
#   y_plot = y[0:1000]
#   print('x shape, y shape', x.shape, y.shape)
#   plotDecisionBoundary(x, y, clf.predict, values, title = '')



      


############################
# Optimize hyperparameters
############################
# train_accs = []
# val_accs = []
# test_accs = []
# ## SVM with RBF kernel
# for t_steps in [25]:
# # for t_steps in [2,10,25,50]:
#   # Read training data
#   x = -1 * np.ones((0, dim * t_steps))
#   y = -1 * np.ones((0,))

#   for i in range(1,2):
#     train_file = 'clusters_'+str(i)
#     data = Data(train_file, verbose=False, _t_steps=t_steps)
#     x_i,y_i = data.get_XY()
    
#     # y element of {0,1}
#     x = np.append(x, x_i, axis=0)
#     y = np.append(y, y_i, axis=0)

#   # Get val and test data
#   val_file = 'clusters_2'
#   val_data = Data(val_file, verbose=False, _t_steps=t_steps)
#   x_val,y_val = val_data.get_XY()

#   test_file = 'clusters_3'
#   test_data = Data(test_file, verbose=False, _t_steps=t_steps)
#   x_test,y_test = test_data.get_XY()
#   for C in [10]:
#   # for C in np.logspace(-3,3,7):
#     gamma = 'auto'
#     kernel = 'rbf'
#     clf = SVM(C, gamma, kernel)
#     clf.train(x,y)

#     train_acc = clf.score(x, y)
#     train_accs.append(train_acc)
#     val_acc = clf.score(x_val, y_val)
#     val_accs.append(val_acc)
#     test_acc = clf.score(x_test, y_test)
#     test_accs.append(test_acc)
#     print "t_steps: %i, C: %f, Train: %.3f, Val: %.3f, Test: %.3f" %(t_steps, C, train_acc,val_acc,test_acc)



