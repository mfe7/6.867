import numpy as np

def relu(x):
    return max(0,x)
def relu_derivative(x):
    f_prime = np.zeros_like(x)
    pos_inds = np.where(x >= 0)[0]
    f_prime[pos_inds] = 1
    return f_prime
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

class NeuralNetwork:
    def __init__(self, num_hidden_layers, hidden_layer_sizes, activation=relu, activation_derivative=relu_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.weights = []
        assert(num_hidden_layers == len(hidden_layer_sizes))
        for i in range(num_hidden_layers-1):
            self.weights.append(np.zeros((hidden_layer_sizes[i], hidden_layer_sizes[i+1])))
            self.biases.append(0)

    def train(self, x, y):
        a_list = []
        z_list = []
        delta_list = []

        # Input vector
        a_list.append(x)

        # Feedforward
        for l in range(num_hidden_layers):
            z = np.dot(self.weights[l].T,a[l-1])+self.biases[l-1]
            z_list.append(z)
            a_list.append(self.activate(z))

        # Output layer
        # todo

        # Backprop
        for l in range(num_hidden_layers-1,1):
            # D = # todo
            delta = np.dot(D, np.dot(self.weights[l+1], delta_list[l+1]))
            delta_list.append(delta)

        # Final gradients
        dloss_dWl = a





if __name__ == '__main__':
    num_hidden_layers = 1
    hidden_layer_sizes = [10]
    input_size = 5
    output_size = 4
    nn = NeuralNetwork(num_hidden_layers, hidden_layer_sizes)






