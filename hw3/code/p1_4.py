import numpy as np
import pylab as pl
from plot_boundary import plotDecisionBoundary
import nn

if __name__ == '__main__':
    # parameters
    name = '4'
    # load data from csv files
    train = pl.loadtxt('data/data'+name+'_train.csv')
    X_train = train[:,0:2]
    Y_train = train[:,2:3]
    validate = pl.loadtxt('data/data'+name+'_validate.csv')
    X_val = validate[:,0:2]
    Y_val = validate[:,2:3]
    test = pl.loadtxt('data/data'+name+'_test.csv')
    X_test = test[:,0:2]
    Y_test = test[:,2:3]

    Y_train[Y_train < 0] = 0
    Y_train[Y_train > 0] = 1
    Y_val[Y_val < 0] = 0
    Y_val[Y_val > 0] = 1
    Y_test[Y_test < 0] = 0
    Y_test[Y_test > 0] = 1

    num_hidden_layers = 1
    hidden_layer_sizes = [100]
    input_size = 2
    output_size = 3
    network = nn.NeuralNetwork(num_hidden_layers, hidden_layer_sizes, input_size, output_size)
    learning_rate = 1e-4
    num_epochs = 50
    prev_acc = 100.0
    acc = 0.0
    num_epochs_used = 0
    num_epochs = 1
    loss = 0.0
    prev_loss = 100.0
    while abs(loss - prev_loss) > 0.05:
        prev_loss = loss
        prev_acc = acc
        num_epochs_used += 1
        network.train(X_train, Y_train, learning_rate, num_epochs)

        y_one_hot = np.eye(3)[np.asarray(Y_val, dtype = np.int32).reshape(-1)]
        
        loss = 0.0
        for i in range(len(y_one_hot)):
            pred = network.output(X_val[i,:])
            loss += network.compute_loss(pred, y_one_hot[i])
        # print loss
        # print "Accuracy: %.2f, Loss: %.2f" %(acc, loss)

    _, train_acc = network.predict(X_train, Y_train)
    _, test_acc = network.predict(X_test, Y_test)
    print "Num Epochs: %i, Training Accuracy: %.2f, Test Acc: %.2f" %(num_epochs_used, train_acc, test_acc)



    plotDecisionBoundary(X_test, np.asarray(Y_test, dtype = np.int32).reshape(-1), 
                         network.evaluate, [0])