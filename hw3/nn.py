import numpy as np
import pylab as pl
from plot_boundary import plotDecisionBoundary


def relu(x):
    return np.maximum(x,0)
def relu_derivative(x):
    x[x <= 0] = 0
    x[x > 0]  = 1
    return x
    # f_prime = np.zeros_like(x)
    # pos_inds = np.where(x >= 0)[0]
    # f_prime[pos_inds] = 1
    # return f_prime
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))


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
        # for i in range(num_hidden_layers+1):
        #     self.weights[i] = np.random.randn(self.layer_dims[i], self.layer_dims[i+1])
        #     self.biases[i] = np.random.randn(self.layer_dims[i+1],)

        self.weights[0] = np.array([[-0.18707593, -0.57116364,  0.87135927,  1.51335649,  0.80079141, -0.41440098, -1.94605898, -0.97956893, -0.79308643, -0.5750369 ],
       [ 0.36597764, -0.54422837, -1.40925812,  0.96691155, -0.21109793,
        -2.05632661, -0.51741284,  0.43944206,  0.46601536, -1.81503629]])
        self.biases[0] = np.array([-0.89934916, -1.35155851, -0.38287867, -0.33701472, -0.02804471,
        0.53519213, -1.75331159,  0.69891045, -0.9670445 ,  0.71366188])

        self.weights[1] = np.array([[-1.28987326,  1.39837888,  0.50437513, -2.20624328, -1.55387087,
        -1.32218096, -0.40354477,  0.21546654, -0.3680378 ,  0.85765001],
       [ 1.08007656, -0.86861468,  1.84695065, -0.16617142, -1.43211465,
        -0.54037974,  1.8211409 ,  0.75250924,  0.40492886, -1.11700955],
       [ 0.90093078,  0.85775522, -0.66586218,  0.08596294, -1.03174721,
         0.83973498, -0.93687592, -0.23966004,  0.04836641, -1.13613744],
       [ 0.25518709,  0.66408231, -0.9280535 , -0.22093368, -1.00523533,
         0.03960567, -1.44661101, -0.49761997,  0.39369518, -0.64744496],
       [ 0.6338689 , -0.72143445, -0.49895748,  2.61994541, -1.23776834,
         0.31327362, -2.77666055, -2.16327754,  1.54424271, -1.70806264],
       [ 0.33025282,  0.67221474, -1.17670102, -0.45418023,  0.23363375,
        -0.00356899, -1.08107921, -1.3746454 , -0.17313848, -0.20522853],
       [-0.6888309 , -0.26560641, -0.96319471, -0.37381552, -0.54884788,
        -0.43070428,  0.61388075, -0.76478669,  0.02974509, -0.62472002],
       [-1.10579369,  0.24268272, -1.82009029, -0.42246684, -0.81930996,
        -0.7903213 ,  0.02734107,  0.58407016, -0.73252182, -0.71919669],
       [ 0.05787481,  1.50249842,  0.17338539,  0.43824447,  0.18908281,
        -0.27597931, -2.76837947,  1.13504499, -0.2265604 ,  1.96139435],
       [-0.68269183, -2.25128425, -0.30627564,  1.3939955 ,  0.8006315 ,
         1.26958524,  1.03730325, -0.77607205, -2.31838554,  0.02741427]])
        self.biases[1] = np.array([ 1.01729498, -0.91845243, -0.15580256, -0.60366744, -1.78563806,
        1.73058401, -1.69949967,  0.43722022, -0.77026456, -1.64112109])
        self.weights[2] = np.array([[ 0.11097127,  0.62009183,  1.55264046],
       [ 0.71411235,  0.38092189, -0.38550374],
       [ 0.03509776, -0.69431843,  0.31009602],
       [ 0.96364519, -0.57146003, -0.06855003],
       [ 0.32698633,  2.3726558 , -1.24525144],
       [-1.57780605,  0.16252072,  0.30338679],
       [ 0.96217567, -1.13213499, -0.07836051],
       [-0.01820266, -1.69982481, -0.83264179],
       [ 1.13368156,  0.30747631, -0.09585687],
       [-0.85553489,  1.32797054, -0.90821894]])
        self.biases[2] = np.array([-0.31317156, -0.93532365,  0.58232589])

    def feedforward(self, x):
        a_list = [None] * (self.num_hidden_layers+2)
        z_list = [None] * (self.num_hidden_layers+1)

        # Input vector
        a_list[0] = x

        # Feedforward
        for l in range(self.num_hidden_layers+1):
            z = np.dot(self.weights[l].T,a_list[l])+self.biases[l]
            z_list[l] = z
            a_list[l+1] = self.activation(z)
        a_list[-1] = softmax(z_list[-1])
        output = a_list[-1]
        return a_list, z_list, output

    def predict(self, x_data, y_data):
        num_pts = np.shape(x_data)[0]
        predictions = np.zeros((num_pts))
        for i in range(num_pts):
            _, _, pred = self.feedforward(x_data[i])
            predictions[i] = np.argmax(pred)
        errors = np.sum(np.squeeze(y_data) != predictions)
        accuracy = 100*(1-errors/float(num_pts))
        return predictions, accuracy

    def evaluate(self, x):
        _, _, pred = self.feedforward(x)
        return np.argmax(pred)

    def train(self, x_data, y_data, learning_rate, num_epochs):
        num_pts = np.shape(y_data)[0]
        y_one_hot = np.eye(3)[np.asarray(y_data, dtype = np.int32).reshape(-1)]
        # y_one_hot = np.zeros((num_pts, self.output_size))
        # y_one_hot[np.arange(num_pts), np.squeeze(y_data)] = 1

        count = 0
        for epoch in range(num_epochs):
            loss = 0
            for x,y in zip(x_data, y_one_hot):
                count += 1
                # print "Training data pt %i of %i (%s,%s)..." %(count, num_pts, x, y)

                a_list, z_list, output = self.feedforward(x)
                delta_list = [None] * (self.num_hidden_layers+1)

                # Output layer
                delta_list[-1] = a_list[-1] - y
                # dl_daL = -np.sum(a_list[-1] - y)
                # daL_dzL = a_list[-1]*(1-a_list[-1])
                # delta_list[-1] = np.multiply(dl_daL, daL_dzL)

                # Backprop
                for l in range(self.num_hidden_layers,0,-1):
                    D = np.diag(self.activation_derivative(z_list[l-1]))
                    delta = np.dot(D, np.dot(self.weights[l], delta_list[l]))
                    delta_list[l-1] = delta

                # Final gradients
                for l in range(0,self.num_hidden_layers+1):
                    a = np.expand_dims(a_list[l], axis=1)
                    d = np.expand_dims(delta_list[l], axis=1)
                    dloss_dWl = np.dot(a, d.T)
                    # print dloss_dWl
                    dloss_dbl = delta_list[l]
                    self.weights[l] -= learning_rate * dloss_dWl
                    self.biases[l] -= learning_rate * dloss_dbl

                loss += self.compute_loss(output, y)
            print "Epoch %i: Loss = %.2f" %(epoch, loss)
    def compute_loss(self, pred, y):
        return -np.dot(np.log(pred), y)


def load_data():
    data = np.loadtxt('hw3_resources/data/data_3class.csv')
    n_training_pts = 300
    n_validation_pts = 250
    n_test_pts = 250
    train_ind = n_training_pts
    val_ind = n_training_pts+n_validation_pts
    test_ind = n_training_pts+n_validation_pts+n_test_pts
    
    x = np.array(data[:,0:2])
    y = np.array(data[:,2:3])

    x_train = x[:train_ind,:]
    x_val = x[train_ind:val_ind,:]
    x_test = x[val_ind:test_ind,:]
    y_train = y[:train_ind,:]
    y_val = y[train_ind:val_ind,:]
    y_test = y[val_ind:test_ind,:]
    return x_train, y_train, x_val, y_val, x_test, y_test



if __name__ == '__main__':
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()

    num_hidden_layers = 2
    hidden_layer_sizes = [10, 10]
    input_size = 2
    output_size = 3
    nn = NeuralNetwork(num_hidden_layers, hidden_layer_sizes, input_size, output_size)
    learning_rate = 1e-2
    num_epochs = 50
    nn.train(x_train, y_train, learning_rate, num_epochs)
    _, accuracy = nn.predict(x_val, y_val)
    print accuracy

    predictions, accuracy = nn.predict(x_train, y_train)
    plotDecisionBoundary(x_train, np.asarray(y_train, dtype = np.int32).reshape(-1), 
                         nn.evaluate, [0, 1, 2])





