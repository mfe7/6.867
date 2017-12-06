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

    n_samples = x.shape[0]
    self.dim = 2

    print('x shape', x.shape)
    print('n_samples,dim', n_samples)
    # Elements from 1D vector x[0] match 2D vector x_new[0]
    x_new = np.array([x[:, ::2], x[:, 1::2]])
    x_new = np.rollaxis(x_new, 1)
    x_new = np.rollaxis(x_new, 2)
    # Unstack along time to get a list of '_t_steps' tensors of shape (n_samples, dim)
    #x_new = tf.unstack(x_new, self._t_steps, 2)

    x_tf1 = tf.placeholder(tf.float32, [self._t_steps, n_samples, self.dim])
    print('xtf1',x_tf1)

    x_tf2= tf.convert_to_tensor(x_new, np.float32)
    print('xtf2', x_tf2)
    return x_tf2


  # x should be formatted
  # for basic rnn: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
  # for lstm: https://www.tensorflow.org/tutorials/recurrent
  def train(self, x, y):

    n_samples = x.shape[0]
    self.batch_size = n_samples # how many trajectories are in the vector x?
    self.lstm_size = 32 # num hidden layers of features
    
    self.x = self.format_x(x)

    self.lstm = rnn.BasicLSTMCell(self.lstm_size, forget_bias = 1.0) # bias forget layer for faster direct gradient pass and faster initial training
    # Initial state of the LSTM memory.
    print('state_size', self.lstm.state_size)
    self.hidden_state = tf.zeros([self.batch_size, self.lstm.state_size])
    self.current_state = tf.zeros([self.batch_size, self.lstm.state_size])
    state = hidden_state, current_state
    probabilities = []
    loss = 0.0
    
'''
  def score(self, x, y):
    acc = self._svm.score(x, y)
    return acc





# one batch correlates to many words = many sniplets
# 

for current_batch_of_snips in snips_in_dataset:
    # The value of state is updated after processing each batch of words.
    output, state = lstm(current_batch_of_snips, state)

    # The LSTM output can be used to make next word predictions
    logits = tf.matmul(output, softmax_w) + softmax_b
    probabilities.append(tf.nn.softmax(logits))
    loss += loss_function(probabilities, target_words)

t=0  t=1    t=2  t=3     t=4
[The, brown, fox, is,     quick]
[The, red,   fox, jumped, high]

s
batch_size = 2, time_steps = 5
The basic pseudocode is as follows:

words_in_dataset = tf.placeholder(tf.float32, [time_steps, batch_size, num_features])
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
# Initial state of the LSTM memory.
hidden_state = tf.zeros([batch_size, lstm.state_size])
current_state = tf.zeros([batch_size, lstm.state_size])
state = hidden_state, current_state
probabilities = []
loss = 0.0
for current_batch_of_words in words_in_dataset:
    # The value of state is updated after processing each batch of words.
    output, state = lstm(current_batch_of_words, state)

    # The LSTM output can be used to make next word predictions
    logits = tf.matmul(output, softmax_w) + softmax_b
    probabilities.append(tf.nn.softmax(logits))
    loss += loss_function(probabilities, target_words)

final_state = state

'''