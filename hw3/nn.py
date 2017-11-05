import numpy as np

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

    def train(self, x, y, learning_rate):
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
        delta_list[-1] = a_list[-1] - y

        print "a_list:", a_list
        print "z_list:", z_list
        print "delta_list:", delta_list
        # Backprop
        for l in range(self.num_hidden_layers,0,-1):
            print "l=",l
            D = np.diag(self.activation_derivative(z_list[l]))
            print D
            print self.weights[l]
            print delta_list[l]
            print np.dot(self.weights[l+1], delta_list[l+1])
            delta = np.dot(D, np.dot(self.weights[l], delta_list[l]))
            delta_list[l] = delta

        # Final gradients
        print a_list
        print delta_list
        for l in range(0,L):
            dloss_dWl = a_list[l]*delta_list[l]
            dloss_dbl = delta_list[l]
            self.weights[l] += -learning_rate * dloss_dWl
            self.biases[l] += -learning_rate * dloss_dbl
        # print self.weights
        # print self.biases





if __name__ == '__main__':
    num_hidden_layers = 1
    hidden_layer_sizes = [10]
    input_size = 5
    output_size = 4
    nn = NeuralNetwork(num_hidden_layers, hidden_layer_sizes, input_size, output_size)
    x = np.array([1,2,3,4,5])
    y = np.array([1,0,0,0])
    learning_rate = 0.1
    nn.train(x, y, learning_rate)




