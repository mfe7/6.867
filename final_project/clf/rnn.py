# FROM: https://www.tensorflow.org/tutorials/recurrent
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import math
import matplotlib.pyplot as plt

class RNN:
  def __init__(self, _t_steps=10, input_dim = 2, n_hidden = 128, n_classes=1, lr=0.0001, batch_size=100, max_iter=1000, max_epochs=3):
    self._lstm = None
    self.input_dim = input_dim
    self.n_hidden = n_hidden # num hidden layers = num features
    self.n_classes = n_classes # binary: cross / no-cross, not one-hot encoded, {0,1} instead 
    
    self.lr = lr # Learning rate
    self.batch_size = batch_size
    self.max_iter = max_iter
    self.max_epochs = max_epochs
    
    # Data params
    self._t_steps = _t_steps # length of trajectory sniples
    self.n_samples = None # set in initialize_graph

    self._init_rnn()

  def _init_rnn(self):
    print('Brei')

    
  # Reshape x from alternating x1x2 to tf placeholder and np data array
  def format_x(self,x):

    self.n_samples = x.shape[0]

    # Elements from 1D vector x[0] match 2D vector x_new[0]
    x_new = np.array([x[:, ::2], x[:, 1::2]])
    x_new = np.rollaxis(x_new, 1)
    x_new = np.rollaxis(x_new, 2)
    # Unstack along time to get a list of '_t_steps' tensors of shape (n_samples, dim)
    #x_new = tf.unstack(x_new, self._t_steps, 2)

    x_tf_data = tf.convert_to_tensor(x_new, np.float32)

    return x_new

  # y of shape (n_samples,)
  def format_y(self,y):

    #y_tf_data = tf.convert_to_tensor(y, np.float32)
    #y_tf_data = tf.reshape(y_tf_data, (self.n_samples, 1))
    y_new = y.reshape((self.n_samples, 1))
    return y_new

  # x should be formatted
  # for basic rnn: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
  # for lstm: https://www.tensorflow.org/tutorials/recurrent
  def initialize_graph(self):
    # This is a single-stack LSTM

    # Define x and y palaceholders 
    self.x = tf.placeholder(tf.float32, [self._t_steps, self.batch_size, self.input_dim])
    self.y = tf.placeholder(tf.float32, [self.batch_size, 1])

    # Define weights (only of output layer)
    # This is for binary classification.
    self.weights = {
      'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
      #'out': tf.Variable(tf.zeros([self.n_hidden, self.n_classes]))
    }
    self.bias = {
      'out': tf.Variable(tf.random_normal([self.n_classes]))
      #'out': tf.Variable(tf.zeros([self.n_classes]))
    }

    ## Initialize graph:
    # Define lstm cell
    self.lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias = 1.0) # bias forget layer for faster direct gradient pass and faster initial training
    
    # Unstack along time to get a list of '_t_steps' tensors of shape (n_samples, dim)
    x_seq = tf.unstack(self.x, num=self._t_steps, axis=0)

    # Get lstm cell output, passes x here!
    # rnn.static_rnn(cell, inputs, initial_state=None, dtype=None, sequence_length = None, scope=None)
    # Creates the RNN defined by "cell" parameter
    self.outputs, self.states = rnn.static_rnn(cell = self.lstm_cell, inputs = x_seq, dtype = tf.float32)

    # Return output layer linear activation
    self.logits= tf.matmul(self.outputs[-1], self.weights['out']) + self.bias['out']
    
    # Predict with logistic (binary) instead of softmax(multiclass)
    #self.prediction = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits)
    self.prediction = tf.sigmoid(x=self.logits)

    # Define loss and optimizer
    # Use mean square error instead of softmax cross entropy
    self.loss = tf.losses.mean_squared_error(labels = self.y, predictions = self.prediction)
    #self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y))
    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
    self.train_op = self.optimizer.minimize(self.loss)

    # Evaluate model
    correct_pred = tf.equal(tf.round(self.prediction), self.y)
    #correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.y, 1))
    # Set accuracy to the mean of elements
    self.acc = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    # Initialize tf variables
    self.init = tf.global_variables_initializer() # returns Op that initializes the global_variables list

  def train_clf(self, x, y):

    print('[STATUS] Start training')
    # Start training:
    self.n_samples = x.shape[0]
    self.x_data = self.format_x(x)
    self.y_data = self.format_y(y)

    self.sess = tf.Session()

    # Run initializer
    self.sess.run(self.init)
    n_iter = int(math.floor(self.n_samples/self.batch_size))

    for epoch in range(self.max_epochs):
      accs = np.zeros(n_iter)
      acc_mean = 0.0

      for step in range(1, self.max_iter+1):
        # Cuts of last piece of trajectory set, which does not fit the batch_size
        if (step*self.batch_size + self.batch_size) < self.n_samples:
          
          # TODO 
          batch_x = self.x_data[:, (step*self.batch_size):(step*self.batch_size + self.batch_size), :]
          batch_y = self.y_data[step:step+self.batch_size, :]
        else:
          break        
        # Run graph
        # Omit optional feed_dict parameter for now
        self.sess.run(self.train_op, feed_dict = {self.x: batch_x, self.y: batch_y})

        # Calculate batch loss & acc
        #loss_tmp, acc_tmp = self.sess.run([self.loss, self.acc], feed_dict = {self.x: batch_x, self.y: batch_y})
        loss_tmp, acc_tmp, pred_tmp = self.sess.run([self.loss, self.acc, self.prediction], feed_dict = {self.x: batch_x, self.y: batch_y})
        accs[step] = acc_tmp

        # print("Step " + str(step) + ", Minibatch Loss= " + \
        #           "{:.4f}".format(loss_tmp) + ", Training Accuracy= " + \
        #           "{:.3f}".format(acc_tmp))
        # print('Pred: {}, Truth: {}'.format(pred_tmp, batch_y))

      acc_mean = np.sum(accs)/n_iter
      # Get full accuracy
      #loss_tmp, acc_tmp = self.sess.run([self.loss, self.acc], feed_dict = {self.x: self.x_data, self.y: self.y_data})
      print('[STATUS] Train accuracy of {}% after epoch {}'.format(acc_mean*100, epoch))
        

  def score(self, x, y):
    n_samples = x.shape[0]
    
    self.x_data = self.format_x(x)
    self.y_data = self.format_y(y)
    
    accs = np.zeros(n_samples)
    acc_mean = 0.0
    
    n_iter = int(math.floor(self.n_samples/self.batch_size))

    for i in range(n_iter):
      #x_i = self.x_data[:, i:(i + 1), :]
      #y_i = self.y_data[i:i+1, :]
      
      x_i = self.x_data[:, (i*self.batch_size):(i*self.batch_size + self.batch_size), :]
      y_i = self.y_data[i:i+self.batch_size, :]

      acc_i = self.sess.run([self.acc], feed_dict = {self.x: x_i, self.y: y_i})
      accs[i] = acc_i[0]
    acc_mean = np.sum(accs)/n_iter
    return acc_mean

  ############################################
  ### Init RNN for trajectory prediction task
  ############################################
  def init_pred_graph(self):
    # This is a single-stack LSTM

    # The target vector y is x_t+1. Hence, x and y have same dimension
    self.x = tf.placeholder(tf.float32, [self._t_steps, self.batch_size, self.input_dim])
    self.y = tf.placeholder(tf.float32, [self._t_steps, self.batch_size, self.input_dim])

    # output_dim = input_dim
    output_dim = self.input_dim
    self.weights = {
      'out': tf.Variable(tf.random_normal([self.n_hidden, output_dim]))
    }
    self.bias = {
      'out': tf.Variable(tf.random_normal([output_dim]))
    }

    ## Initialize graph:
    # Define lstm cell
    self.lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias = 1.0) # bias forget layer for faster direct gradient pass and faster initial training

    # Initialize initial state 
    init_state = self.lstm_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
    self.state_in = tf.identity(init_state, name='state_in') # tf identity copies the vector

    # Unstack along time to get a list of '_t_steps' tensors of shape (n_samples, dim)
    x_seq = tf.unstack(self.x, num=self._t_steps, axis=0)

    # Define rnn decoder
    # loop_function applies fct(y_i) for input x_i+1
    # scope = VariableScope for created subgraph, default='rnn_decoder'
    # Compute target output / prediction vector and hidden cell state. Can be same for RNN, but different for LSTM
    # self.outputs.shape = 1,128
    self.outputs, self.states = tf.contrib.legacy_seq2seq.rnn_decoder(decoder_inputs=x_seq, initial_state=init_state, cell=self.lstm_cell, loop_function=None)

    # Return output layer linear activation
    # Logits are of shape (1,2) and predict x_t+1
    self.logits= tf.matmul(self.outputs[-1], self.weights['out']) + self.bias['out']

    # Predict 
    self.pred = self.logits

    # Loss is LMSE between predicted logit and x_t+1
    self.loss = tf.losses.mean_squared_error(labels = self.y[-1], predictions = self.pred)
    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
    self.train_op = self.optimizer.minimize(self.loss)

    # Initialize tf variables
    self.init = tf.global_variables_initializer() # returns Op that initializes the global_variables list

  def train_pred(self, x):

    print('[STATUS] Start training')
    # Start training:
    self.n_samples = x.shape[0]
    self.x_data = self.format_x(x)

    # Make y vector be a 1 time-step shifted x vector with x_init = x[0]
    # TODO: HOW SHOULD I label the last predicted value, because I do not have that value?
    self.y_data = self.x_data[:,:,:]
    self.y_data[:,:-1,:] = self.x_data[:,1:,:]
    self.y_data[:,-1,:] = self.x_data[:,-1,:]

    self.sess = tf.Session()

    # Run initializer
    self.sess.run(self.init)
    n_iter = int(math.floor(self.n_samples/self.batch_size))

    for epoch in range(self.max_epochs):
      #accs = np.zeros(n_iter)
      #acc_mean = 0.0
      loss_total = 0.0
      for step in range(1, self.max_iter+1):
        # Cuts of last piece of trajectory set, which does not fit the batch_size
        if (step*self.batch_size + self.batch_size) < self.n_samples:
          
          # TODO 
          batch_x = self.x_data[:, (step*self.batch_size):(step*self.batch_size + self.batch_size), :]
          batch_y = self.y_data[:, (step*self.batch_size):(step*self.batch_size + self.batch_size), :]

        else:
          break        
        # Run graph
        # Omit optional feed_dict parameter for now
        self.sess.run(self.train_op, feed_dict = {self.x: batch_x, self.y: batch_y})

        # Calculate batch loss & pred
        loss_tmp, y_true, y_pred = self.sess.run([self.loss, self.y, self.pred], feed_dict = {self.x: batch_x, self.y: batch_y})
        loss_total += loss_tmp

      print('[STATUS] Total loss of {} after epoch {}'.format(loss_total, epoch))
        
  def score_pred(self, x):
    n_samples = x.shape[0]
    self.x_data = self.format_x(x)

    # Make y vector be a 1 time-step shifted x vector with x_init = x[0]
    self.y_data = self.x_data[:,:,:]
    self.y_data[:,:-1,:] = self.x_data[:,1:,:]
    self.y_data[:,-1,:] = self.x_data[:,-1,:]

    loss_total = 0.0
    
    t_predict = 10 # Length of predicted trajectory
    plot_one_future_t_step = False # Plot only one step in the future
    
    n_iter = int(math.floor(self.n_samples/self.batch_size))

    for i in range(n_iter):
      #x_i = self.x_data[:, i:(i + 1), :]
      #y_i = self.y_data[i:i+1, :]
      
      x_i = self.x_data[:, (i*self.batch_size):(i*self.batch_size + self.batch_size), :]
      y_i = self.y_data[:, (i*self.batch_size):(i*self.batch_size + self.batch_size), :]

      loss_tmp, y_true, y_pred = self.sess.run([self.loss, self.y, self.pred], feed_dict = {self.x: x_i, self.y: y_i})
      loss_total += loss_tmp
      

      if i > 150: # give him some time for whatever

        # Plot onyl one step in the future
        if plot_one_future_t_step:
          plt.plot(y_true[:,:,0], y_true[:,:,1], '-', color = 'green')
          plt.plot(y_pred[0,0], y_pred[0,1], 'o', color = 'red')
          print_i_steps = 1
          if i%print_i_steps == 0: # print every ... steps
            plt.show()

        ## Predict by feeding itself

        y_preds = np.zeros((t_predict, 1, self.input_dim))
        for t_i in range(t_predict):
          # Shift everything one index into the future
          x_i[:, :-1,:] = x_i[:, 1:,:]
          x_i[:, -1, :] = y_pred[0]
          # TODO assign useful values to y_i[-1]
          y_i[:, :-1,:] = y_i[:, 1:,:]
          y_i[:, -1, :] = y_pred[0]
          # Predict
          y_past, y_pred = self.sess.run([self.y, self.pred], feed_dict = {self.x: x_i, self.y: y_i})
          
          y_preds[t_i,:,:] = y_pred[:,:]
        
        print('[STATUS] Print predicted test trajectory (red) for given trajectory (green) for sniplet {}'.format(i))
        plt.plot(y_true[:,:,0], y_true[:,:,1], '-', color = 'green')
        plt.plot(y_preds[:,0,0], y_preds[:,0,1], '-', color = 'red')
        plt.show()


    return loss_total



## Template code for word classification:

# # one batch correlates to many words = many sniplets
# # 

# for current_batch_of_snips in snips_in_dataset:
#     # The value of state is updated after processing each batch of words.
#     output, state = lstm(current_batch_of_snips, state)

#     # The LSTM output can be used to make next word predictions
#     logits = tf.matmul(output, softmax_w) + softmax_b
#     probabilities.append(tf.nn.softmax(logits))
#     loss += loss_function(probabilities, target_words)

# t=0  t=1    t=2  t=3     t=4
# [The, brown, fox, is,     quick]
# [The, red,   fox, jumped, high]

# s
# batch_size = 2, time_steps = 5
# The basic pseudocode is as follows:

# words_in_dataset = tf.placeholder(tf.float32, [time_steps, batch_size, num_features])
# lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
# # Initial state of the LSTM memory.
# hidden_state = tf.zeros([batch_size, lstm.state_size])
# current_state = tf.zeros([batch_size, lstm.state_size])
# state = hidden_state, current_state
# probabilities = []
# loss = 0.0
# for current_batch_of_words in words_in_dataset:
#     # The value of state is updated after processing each batch of words.
#     output, state = lstm(current_batch_of_words, state)

#     # The LSTM output can be used to make next word predictions
#     logits = tf.matmul(output, softmax_w) + softmax_b
#     probabilities.append(tf.nn.softmax(logits))
#     loss += loss_function(probabilities, target_words)

# final_state = state

# '''