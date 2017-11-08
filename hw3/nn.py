import numpy as np
import pylab as pl

def relu(x):
    return np.maximum(x,0)
    # return np.clip(x,0,None)
def relu_derivative(x):
    f_prime = np.zeros_like(x)
    pos_inds = np.where(x >= 0)[0]
    f_prime[pos_inds] = 1
    return f_prime
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))
def softmax_derivative(x):
    return None


class NeuralNetwork:
    def __init__(self, num_hidden_layers, hidden_layer_sizes, input_size, output_size, activation=relu, activation_derivative=relu_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.input_size = input_size
        self.output_size = output_size
        self.weights = [None]*(num_hidden_layers+1)
        self.biases = [None]*(num_hidden_layers+1)
        self.num_hidden_layers = num_hidden_layers
        self.layer_dims = [self.input_size] + hidden_layer_sizes + [self.output_size]
        for i in range(num_hidden_layers+1):
            self.weights[i] = np.zeros((self.layer_dims[i], self.layer_dims[i+1]))
            self.biases[i] = np.zeros((self.layer_dims[i+1]))

    def train(self, x_data, y_data, learning_rate):
        num_pts = np.shape(y_data)[0]
        y_one_hot = np.zeros((num_pts, self.output_size))
        y_one_hot[np.arange(num_pts), np.squeeze(y_data)] = 1

        count = 0
        for x,y in zip(x_data, y_data):
            count += 1
            print "Training data pt %i of %i (%s,%i)..." %(count, num_pts, x, y)

            a_list = [None] * (self.num_hidden_layers+2)
            z_list = [None] * (self.num_hidden_layers+1)
            delta_list = [None] * (self.num_hidden_layers+1)

            # Input vector
            a_list[0] = x

            # Feedforward
            for l in range(self.num_hidden_layers+1):
                z = np.dot(self.weights[l].T,a_list[l])+self.biases[l]
                z_list[l] = z
                a_list[l+1] = self.activation(z)
            a_list[-1] = softmax(z_list[-1])

            # Output layer
            dl_daL = -np.sum(a_list[-1] - y)
            daL_dzL = a_list[-1]*(1-a_list[-1])
            delta_list[-1] = np.multiply(dl_daL, daL_dzL)

            print "a_list:", a_list
            print "z_list:", z_list
            print "delta_list:", delta_list
            # Backprop
            for l in range(self.num_hidden_layers,-1,-1):
                print "l=",l

                D = np.diag(self.activation_derivative(z_list[l]))
                delta = np.multiply(D, self.weights[l]*delta_list[l])
                # delta = np.dot(D, np.dot(self.weights[l], delta_list[l]))
                delta_list[l] = delta

            # Final gradients
            print a_list
            print delta_list
            for l in range(0,self.num_hidden_layers+2):
                dloss_dWl = a_list[l]*delta_list[l]
                dloss_dbl = delta_list[l]
                self.weights[l] += -learning_rate * dloss_dWl
                self.biases[l] += -learning_rate * dloss_dbl
            # print self.weights
            # print self.biases


def load_data():
    data = np.loadtxt('hw3_resources/data/data_3class.csv')
    n_training_pts = 300
    n_validation_pts = 250
    n_test_pts = 250
    train_ind = n_training_pts
    val_ind = n_training_pts+n_validation_pts
    test_ind = n_training_pts+n_validation_pts+n_test_pts
    
    x = np.array(data[:,0:2], dtype=int)
    y = np.array(data[:,2:3], dtype=int)

    x_train = x[:train_ind,:]
    x_val = x[train_ind:val_ind,:]
    x_test = x[val_ind:test_ind,:]
    y_train = y[:train_ind,:]
    y_val = y[train_ind:val_ind,:]
    y_test = y[val_ind:test_ind,:]
    return x_train, y_train, x_val, y_val, x_test, y_test



if __name__ == '__main__':
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()

    num_hidden_layers = 1
    hidden_layer_sizes = [10]
    input_size = 2
    output_size = 3
    nn = NeuralNetwork(num_hidden_layers, hidden_layer_sizes, input_size, output_size)
    learning_rate = 0.1
    nn.train(x_train, y_train, learning_rate)





