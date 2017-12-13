import numpy as np
from svm import SVM
from read_data import Data
from rnn import RNN
import tensorflow as tf
from plotBoundary import plotDecisionBoundary 
import matplotlib.pyplot as plt
from gen_data import Data_gen

clf_sel = 'rnn'

plot = False
plot_dec_bound = True # prints dec bound for linear svm, requires t_steps = 1
test_model = True
generate_data = False # generates sinusodial train and test data and predicts traj
gen_fct = 'sin'

dim = 2
t_steps = 50 # trajectory sniplet length

# Read training data
x = -1 * np.ones((0, dim * t_steps))
y = -1 * np.ones((0,))

for i in range(1,3):
  train_file = 'clusters_'+str(i)
  data = Data(train_file, verbose=False, _t_steps=t_steps)
  x_i,y_i = data.get_XY()
  
  # y element of {0,1}
  x = np.append(x, x_i, axis=0)
  y = np.append(y, y_i, axis=0)

# x = [np.array([1, 2]),np.array([2, 3]),np.array([3,4 ]),np.array([4,5])]
# y = [0,1,2,3]
# print(len(x), x[0].shape)
# plt.plot(x[:][0],x[:][1])
# plt.show()


## Save and load txt from file
# np.savetxt("data/clusters_train.csv", x, delimiter=",")
# np.savetxt("data/clusters_train.csv", y, delimiter=",")

# x = np.loadtxt('data/clusters_train.csv', delimiter=',')
# y = np.loadtxt('data/clusters_train.csv', delimiter=',')

eps = np.zeros(20)
loss_n_ep = np.zeros((4, 4))

## Training parameters:
for lr in [0.001]:
  for n_hidden in [512]:
    for batch_size in [100]:
      for max_epochs in [1]:
        #print('[STATUS] LEARNING RATE = {}'.format(lr))
        print('[STATUS] N hidden units = {}'.format(n_hidden))
        #print('[STATUS] BATCH SIZE = {}'.format(batch_size))

        # Generate test data of linear, sinusoidal, nonlinear shape:
        # Only generate x1, x2 position - no label
        n_samples = 50000
        if generate_data:
          data_gen = Data_gen(_t_steps=t_steps, input_dim=dim)
          x = data_gen.gen_data(fct=gen_fct, n_samples=n_samples)


          # Plot generated data
          # for n in range(100):
          #   plt.plot(x[n, ::2], x[n, 1::2])
          # plt.xlabel('x1')
          # plt.ylabel('x2')
          # plt.title('Sinusoidal trajectories')
          # plt.xlim([-25, 25])
          # plt.ylim([-25, 25])
          # plt.show()
              
        ## SVM with RBF kernel
        if clf_sel == 'svm_rbf':
          C = 1
          gamma = 'auto'
          kernel = 'rbf'
          clf = SVM(C, gamma, kernel)

          clf.train(x,y)

          train_acc = clf.score(x, y)
          print('[STATUS] Train accuracy of {0:.2f}%'.format(train_acc*100))

          if plot_dec_bound == True:
            values = [0,1]
            x_plot = x[0:1000,:]
            y_plot = y[0:1000]
            print('x shape, y shape', x.shape, y.shape)
            plotDecisionBoundary(x, y, clf.predict, values, title = '')


        ## RNN
        elif clf_sel == 'rnn':

          input_dim = 2 # x and y position of pedestrian
          #n_hidden = 128 # num hidden layers = num features
          n_classes = 1 # binary: cross / no-cross, not one-hot encoded, {0,1} instead 
          #max_epochs = 60 # reduce this for faster runtime

          #lr = 0.01 # Learning rate
          #batch_size = 100
          max_iter = 10000

          tf.reset_default_graph()
          clf = RNN(t_steps, input_dim, n_hidden, n_classes, lr, batch_size, max_iter, max_epochs)
          
          rnn_classify = False

          print('[STATUS] Start training {} to classify {} with input_dim={}, n_hidden={}, max_epochs={}, lr={}, batch_size={}, max_iter={}'.format(clf_sel, 
            rnn_classify, input_dim, n_hidden, max_epochs, lr, batch_size, max_iter))
          print('x, y shape', x.shape, y.shape)
          if rnn_classify == True:
            clf.initialize_graph()
            clf.train_clf(x, y)
          # Create graph for trajectory prediction
          else:
            print('[STATUS] Train trajectory predictor for {} epochs'.format(max_epochs))
            clf.init_pred_graph()
            clf.train_pred(x)

          print('[STATUS] Training finished')

        ## Test
        if test_model == True:
          # Get val and test data
          val_file = 'clusters_2'
          val_data = Data(val_file, verbose=False, _t_steps=t_steps)
          x_val,y_val = val_data.get_XY()

          test_file = 'clusters_3'
          test_data = Data(test_file, verbose=False, _t_steps=t_steps)
          x_test,y_test = test_data.get_XY()
          if generate_data == True:
            n_tst_samples = 1000
            x_test = data_gen.gen_data(fct=gen_fct, n_samples=n_tst_samples)
            # Take training data, to see if i overfit?

          # Test and predict RNN trajectories
          if clf_sel == 'rnn' and rnn_classify == False:  
            test_loss, plot_trajs = clf.score_pred(x_test, plot_pred=True)
            avg_loss = test_loss/x_test.shape[0]
            avg_batch_loss = test_loss/(x_test.shape[0]/batch_size)
            print('[STATUS] Total test loss of {0:.2f}, avg batch loss {1:.5f}, avg sample loss {2:.2f}'.format(test_loss,avg_batch_loss, avg_loss))

            # Print predicted trajectories
            print('[STATUS] Print predicted trajectories')
            if generate_data==True:
              n_tst_samples = 4 * batch_size
              x_test = data_gen.gen_data(fct=gen_fct, n_samples=n_tst_samples)
              test_loss, plot_trajs = clf.score_pred(x_test, plot_pred=True)

            y_true_arr = plot_trajs[0]
            y_preds_arr = plot_trajs[1]
            y_pred_future_arr = plot_trajs[2]
            y_pred_true_arr = plot_trajs[3]
            

            for y_t in plot_trajs:
              print(len(y_t))

            for i, (y_true, y_pred, y_pred_fut, y_pred_true) in enumerate(zip(y_true_arr, y_preds_arr, y_pred_future_arr, y_pred_true_arr)):
              y_pred_plt = np.asarray(y_pred)
              plt.plot(y_pred_plt[:,0,0], y_pred_plt[:,0,1], '--', color='blue')
              plt.plot(y_pred_plt[0,0,0], y_pred_plt[0,0,1], 'o', color='blue')
              
              y_t_plt = np.asarray(y_true)
              plt.plot(y_t_plt[:,0,0], y_t_plt[:,0,1], '-', color='green')
              plt.plot(y_t_plt[0,0,0], y_t_plt[0,0,1], 'o', color='green')

              y_f = np.asarray(y_pred_fut)

              plt.plot(y_f[:,0,0], y_f[:,0,1], '-', color='red')
              # if predicted trajectory existing, plot it. Only plots the correct trajectory, if its a sniplet of a longer trajectory
              #plt.plot(y_pred_true[:,:,0], y_pred_true[:,:,1], '--', color = 'blue')

              plt.xlabel('x1')
              plt.ylabel('x2')
              plt.show()

          else:
            val_acc = clf.score(x_val, y_val)
            print('[STATUS] Validation accuracy of {0:.2f}%'.format(val_acc*100))

            test_acc = clf.score(x_test, y_test)
            print('[STATUS] Test accuracy of {0:.2f}%'.format(test_acc*100))
          

        # Plot train, val and test dataset with ground_truth scores
        if plot == True:
          print('[STATUS] Plot train set')
          plot_nsnips = 1000 # plot no more than 1000 trajectory sniplets to reduce plotting time
          data.plot_clf(x,y, plot_nsnips)

          print('[STATUS] Plot validation set]')
          data.plot_clf(x_val,y_val, plot_nsnips)

          print('[STATUS] Plot test set')
          data.plot_clf(x_test,y_test, plot_nsnips)

