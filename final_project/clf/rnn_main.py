import numpy as np
from svm import SVM
from read_data import Data
from rnn import RNN
import tensorflow as tf
from plotBoundary import plotDecisionBoundary 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from gen_data import Data_gen


plot = False
plot_dec_bound = True # prints dec bound for linear svm, requires t_steps = 1
test_model = True
generate_data = False # generates sinusodial train and test data and predicts traj


#########################
# find optimal number of training pts
#########################
# data_pts = []
# test_accs = []
# t_steps = 10 # trajectory sniplet length

# test_file = 'clusters_9'
# test_data = Data(test_file, verbose=False, _t_steps=t_steps)
# x_test,y_test = test_data.get_XY()

# input_dim = 2 # x and y position of pedestrian
# n_hidden = 256 # num hidden layers = num features
# n_classes = 1 # binary: cross / no-cross, not one-hot encoded, {0,1} instead 
# max_epochs = 3 # reduce this for faster runtime

# lr = 0.001 # Learning rate
# batch_size = 5
# max_iter = 10000

# for j in [2,3,4,5,8,22,23,24,25,26,27,28,29]:
#   x = -1 * np.ones((0, input_dim * t_steps))
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

#   tf.reset_default_graph()
#   clf = RNN(t_steps, input_dim, n_hidden, n_classes, lr, batch_size, max_iter, max_epochs)
#   clf.initialize_graph()
#   clf.train_clf(x, y, x_test, y_test)
#   print 'Done training!!'

#   test_acc = clf.score(x_test, y_test)
#   data_pts.append(x.shape[0])
#   test_accs.append(test_acc)
#   print x.shape[0], test_acc
#   print '----'
#   print "data_pts=",data_pts
#   print "test_accs=",test_accs
#   print '----'

# # SVM
# data_pts= [1675, 4067, 6787, 9700, 9700, 9700, 9704, 13229, 15459, 19073, 22841, 22841, 22841]
# test_accs= [0.79572338489535943, 0.83303002729754327, 0.84167424931756141, 0.84758871701546856, 0.84758871701546856, 0.84758871701546856, 0.84758871701546856, 0.85304822565969063, 0.8575978161965423, 0.86396724294813465, 0.86396724294813465, 0.86396724294813465, 0.86396724294813465]
# plt.plot(data_pts,test_accs,'r--o')

# # RNN
# data_pts= [4422, 10793, 17968, 25711, 35109]
# test_accs= [0.76003432, 0.80034304, 0.78542024, 0.79691255, 0.78627789]
# plt.plot(data_pts,test_accs,'b--x')

# plt.legend(['SVM', 'LSTM'],loc='center right')
# plt.xlabel('# Trajectory Snippets')
# plt.ylabel('Test Accuracy')
# plt.show()

#########################
# plots of labeled trajectories
#########################
dim = 2
t_steps = 10 # trajectory sniplet length

x = -1 * np.ones((0, dim * t_steps))
y = -1 * np.ones((0,))
for i in range(1,8) + range(21,29):
    train_file = 'clusters_'+str(i)
    data = Data(train_file, verbose=False, _t_steps=t_steps)
    x_i,y_i = data.get_XY()
    
    # y element of {0,1}
    x = np.append(x, x_i, axis=0)
    y = np.append(y, y_i, axis=0)


input_dim = 2 # x and y position of pedestrian
n_hidden = 256 # num hidden layers = num features
n_classes = 1 # binary: cross / no-cross, not one-hot encoded, {0,1} instead 
max_epochs = 3 # reduce this for faster runtime

lr = 0.001 # Learning rate
batch_size = 5
max_iter = 10000

tf.reset_default_graph()
clf = RNN(t_steps, input_dim, n_hidden, n_classes, lr, batch_size, max_iter, max_epochs)

rnn_classify = False

print('[STATUS] Start training {} to classify {} with input_dim={}, n_hidden={}, max_epochs={}, lr={}, batch_size={}, max_iter={}'.format('rnn', 
  rnn_classify, input_dim, n_hidden, max_epochs, lr, batch_size, max_iter))
print('x, y shape', x.shape, y.shape)
clf.initialize_graph()
clf.train_clf(x, y)

print('[STATUS] Training RNN finished')

test_file = 'clusters_9'
test_data = Data(test_file, verbose=False, _t_steps=t_steps)
x_test,y_test = test_data.get_XY()

false_pos = 0
false_neg = 0
true_pos = 0
true_neg = 0

for i in range(x_test.shape[0]):
  pred = clf.predict(x_test[i].reshape(1,-1))[0][0][0]
  label = y_test[i]
  # print "pred:", pred, ", label:", label

  xs = x_test[i][::2]
  ys = x_test[i][1::2]
  if pred == 0 and label == 0:
    # plt.plot(xs, ys, '-', color = 'green')
    true_neg += 1
  elif pred == 1 and label == 0:
    # plt.plot(xs, ys, '-', color = 'red') 
    false_pos += 1
  elif pred == 0 and label == 1:
    plt.plot(xs, ys, '-', color = 'orange') 
    false_neg += 1
  elif pred == 1 and label == 1:
    plt.plot(xs, ys, '-', color = 'cyan') 
    true_pos += 1

print "false_pos:",false_pos,false_pos/float(x_test.shape[0])
print "false_neg:",false_neg,false_neg/float(x_test.shape[0])
print "true_pos:",true_pos,true_pos/float(x_test.shape[0])
print "true_neg:",true_neg,true_neg/float(x_test.shape[0])
print "test acc:", (true_pos+true_neg)/float(x_test.shape[0])

# green_patch = mpatches.Patch(color='green', label='True negatives')
# red_patch = mpatches.Patch(color='red', label='False positivies')
orange_patch = mpatches.Patch(color='orange', label='False negatives')
cyan_patch = mpatches.Patch(color='cyan', label='True positives')
# plt.legend(handles=[green_patch, red_patch],loc='lower left')
plt.legend(handles=[orange_patch, cyan_patch],loc='lower left')

plt.xlim([-20,20])
plt.ylim([-20,30])

plt.show()


#######################
# hyperparam search
###################
# dim = 2



# # dimensions
# input_dim = 2 # x and y position of pedestrian
# n_classes = 1 # binary: cross / no-cross, not one-hot encoded, {0,1} instead 
# max_iter = 10000 # useless!!!

# # hyperparams
# t_steps = 25 # trajectory sniplet length
# n_hidden = 128 # num hidden layers
# max_epochs = 5 # reduce this for faster runtime
# lr = 1e-3 # Learning rate
# batch_size = 5


# for t_steps in [2,10,25,50]:
#   val_file = 'clusters_8'
#   val_data = Data(val_file, verbose=False, _t_steps=t_steps)
#   x_val,y_val = val_data.get_XY()
  
#   x = -1 * np.ones((0, dim * t_steps))
#   y = -1 * np.ones((0,))
#   for i in range(1,2):
#   # for i in range(1,8) + range(21,29):
#     train_file = 'clusters_'+str(i)
#     data = Data(train_file, verbose=False, _t_steps=t_steps)
#     x_i,y_i = data.get_XY()

#     # y element of {0,1}
#     x = np.append(x, x_i, axis=0)
#     y = np.append(y, y_i, axis=0)
#   for n_hidden in [8,64,128,256]:
#     for batch_size in [1,5,10]:
#       print '--------------'
#       print 'bs:',batch_size, 'n_hidden:',n_hidden,'tsteps:',t_steps
#       tf.reset_default_graph()
#       clf = RNN(t_steps, input_dim, n_hidden, n_classes, lr, batch_size, max_iter, max_epochs)
#       clf.initialize_graph()
#       clf.train_clf(x, y, x_val, y_val)
#       # val_acc = clf.score(x_val,y_val)
#       # print 'bs:',batch_size, 'n_hidden:',n_hidden,'tsteps:',t_steps, 'val:', val_acc
# print('[STATUS] Training RNN finished')

