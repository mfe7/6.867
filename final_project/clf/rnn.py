# FROM: https://www.tensorflow.org/tutorials/recurrent
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class RNN:
  def __init__(self):
    self._lstm = None
    self._init_rnn()

  def _init_rnn(self):
    self._t_steps = 10 # length of trajectory: max, const?
    print('Brei')

  # Reshape x from alternating x1x2 to (n_samples, _t_steps, dim)
  def format_x(self,x):

    self.n_samples = x.shape[0]
    self.dim = 2

    print('x shape', x.shape)
    print('n_samples,dim', self.n_samples)
    # Elements from 1D vector x[0] match 2D vector x_new[0]
    x_new = np.array([x[:, ::2], x[:, 1::2]])
    x_new = np.rollaxis(x_new, 1)
    x_new = np.rollaxis(x_new, 2)
    # Unstack along time to get a list of '_t_steps' tensors of shape (n_samples, dim)
    #x_new = tf.unstack(x_new, self._t_steps, 2)

    # TODO: I think n_sampels should be batch_size here 
    x_tf1 = tf.placeholder(tf.float32, [self._t_steps, self.batch_size, self.dim])

    x_tf_data = tf.convert_to_tensor(x_new, np.float32)


    return x_tf1, x_new

  # y of shape (n_samples,)
  def format_y(self,y):
    y_tf_var = tf.placeholder(tf.float32, [self.batch_size, 1])

    #y_tf_data = tf.convert_to_tensor(y, np.float32)
    #y_tf_data = tf.reshape(y_tf_data, (self.n_samples, 1))
    y_new = y.reshape((self.n_samples, 1))
    return y_tf_var, y_new

  # x should be formatted
  # for basic rnn: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
  # for lstm: https://www.tensorflow.org/tutorials/recurrent
  def initialize_graph(self, x, y):


    self.n_classes = 1 # binary: cross / no-cross, not one-hot encoded, {0,1} instead 
    self.n_hidden = 128 # num hidden layers = num features

    self.n_samples = x.shape[0]
    self.batch_size = 100# n_samples # take full data as batch
    self.lr = 0.0001 # Learning rate
    self.max_iterations = 1000
    
    # Reformat data
    self.x, self.x_data = self.format_x(x)
    self.y, self.y_data = self.format_y(y)

    # Define weights (only of output layer)
    # This is for binary classification.
    self.weights = {
      'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
    }
    self.bias = {
      'out': tf.Variable(tf.random_normal([self.n_classes]))
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
    print('shape y, pred', self.y.shape, self.prediction.shape)
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

    # Start training:
    with tf.Session() as sess:

      # Run initializer
      sess.run(self.init)

      for step in range(1, self.max_iterations+1):
        if (step + self.batch_size) < self.n_samples:
          
          # TODO 
          batch_x = self.x_data[:, step:step+self.batch_size, :]
          batch_y = self.y_data[step:step+self.batch_size, :]
        else:
          break        
        # Run graph
        # Omit optional feed_dict parameter for now
        sess.run(self.train_op, feed_dict = {self.x: batch_x, self.y: batch_y})

        # Calculate batch loss & acc
        loss_tmp, acc_tmp = sess.run([self.loss, self.acc], feed_dict = {self.x: batch_x, self.y: batch_y})

        print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss_tmp) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc_tmp))

      print('Optimization finished')

      # Calculate test accuracy
      # def score(self, x, y):
      # acc = self._svm.score(x, y)
      # return acc



### Earlier tests

    # # Initial state of the LSTM memory.
    # print('state_size', self.lstm.state_size)
    # self.hidden_state = tf.zeros([self.batch_size, self.lstm.state_size])
    # self.current_state = tf.zeros([self.batch_size, self.lstm.state_size])
    # state = hidden_state, current_state
    # probabilities = []
    # loss = 0.0
    



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