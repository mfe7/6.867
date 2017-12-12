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
# plots of labeled trajectories
#########################
# dim = 2
# t_steps = 25 # trajectory sniplet length

# x = -1 * np.ones((0, dim * t_steps))
# y = -1 * np.ones((0,))
# for i in range(1,8) + range(21,29):
#     train_file = 'clusters_'+str(i)
#     data = Data(train_file, verbose=False, _t_steps=t_steps)
#     x_i,y_i = data.get_XY()
    
#     # y element of {0,1}
#     x = np.append(x, x_i, axis=0)
#     y = np.append(y, y_i, axis=0)


# input_dim = 2 # x and y position of pedestrian
# n_hidden = 128 # num hidden layers = num features
# n_classes = 1 # binary: cross / no-cross, not one-hot encoded, {0,1} instead 
# max_epochs = 1 # reduce this for faster runtime

# lr = 0.001 # Learning rate
# batch_size = 5
# max_iter = 10000

# tf.reset_default_graph()
# clf = RNN(t_steps, input_dim, n_hidden, n_classes, lr, batch_size, max_iter, max_epochs)

# rnn_classify = False

# print('[STATUS] Start training {} to classify {} with input_dim={}, n_hidden={}, max_epochs={}, lr={}, batch_size={}, max_iter={}'.format('rnn', 
#   rnn_classify, input_dim, n_hidden, max_epochs, lr, batch_size, max_iter))
# print('x, y shape', x.shape, y.shape)
# clf.initialize_graph()
# clf.train_clf(x, y)

# print('[STATUS] Training RNN finished')

# test_file = 'clusters_9'
# test_data = Data(test_file, verbose=False, _t_steps=t_steps)
# x_test,y_test = test_data.get_XY()

# false_pos = 0
# false_neg = 0
# true_pos = 0
# true_neg = 0

# for i in range(x_test.shape[0]):
#   pred = clf.predict(x_test[i].reshape(1,-1))[0][0][0]
#   label = y_test[i]
#   # print "pred:", pred, ", label:", label

#   xs = x_test[i][::2]
#   ys = x_test[i][1::2]
#   if pred == 0 and label == 0:
#     plt.plot(xs, ys, '-', color = 'green')
#     true_neg += 1
#   elif pred == 1 and label == 0:
#     plt.plot(xs, ys, '-', color = 'red') 
#     false_pos += 1
#   elif pred == 0 and label == 1:
#     # plt.plot(xs, ys, '-', color = 'orange') 
#     false_neg += 1
#   elif pred == 1 and label == 1:
#     # plt.plot(xs, ys, '-', color = 'cyan') 
#     true_pos += 1

# print "false_pos:",false_pos,false_pos/float(x_test.shape[0])
# print "false_neg:",false_neg,false_neg/float(x_test.shape[0])
# print "true_pos:",true_pos,true_pos/float(x_test.shape[0])
# print "true_neg:",true_neg,true_neg/float(x_test.shape[0])
# print "test acc:", (true_pos+true_neg)/float(x_test.shape[0])

# green_patch = mpatches.Patch(color='green', label='True negatives')
# red_patch = mpatches.Patch(color='red', label='False positivies')
# # # orange_patch = mpatches.Patch(color='orange', label='False negatives')
# # # cyan_patch = mpatches.Patch(color='cyan', label='True positives')
# plt.legend(handles=[green_patch, red_patch],loc='lower left')
# # # plt.legend(handles=[orange_patch, cyan_patch],loc='lower left')

# plt.xlim([-20,20])
# plt.ylim([-20,30])

# plt.show()


#######################
# hyperparam search
###################
dim = 2
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

# dimensions
input_dim = 2 # x and y position of pedestrian
n_classes = 1 # binary: cross / no-cross, not one-hot encoded, {0,1} instead 
max_iter = 10000 # useless!!!

# hyperparams
n_hidden = 128 # num hidden layers
max_epochs = 5 # reduce this for faster runtime
lr = 1e-3 # Learning rate
batch_size = 5

tf.reset_default_graph()
clf = RNN(t_steps, input_dim, n_hidden, n_classes, lr, batch_size, max_iter, max_epochs)

rnn_classify = False

print('[STATUS] Start training {} to classify {} with input_dim={}, n_hidden={}, max_epochs={}, lr={}, batch_size={}, max_iter={}'.format('rnn', 
  rnn_classify, input_dim, n_hidden, max_epochs, lr, batch_size, max_iter))
print('x, y shape', x.shape, y.shape)
clf.initialize_graph()
clf.train_clf(x, y)

print('[STATUS] Training RNN finished')

val_file = 'clusters_8'
val_data = Data(val_file, verbose=False, _t_steps=t_steps)
x_val,y_val = val_data.get_XY()
val_acc = clf.score(x_val,y_val)
print "val acc:", val_acc

test_file = 'clusters_9'
test_data = Data(test_file, verbose=False, _t_steps=t_steps)
x_test,y_test = test_data.get_XY()
test_acc = clf.score(x_test,y_test)
print "test acc:", test_acc
