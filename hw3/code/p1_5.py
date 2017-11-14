import numpy as np
import pylab as pl
from plot_boundary import plotDecisionBoundary
import nn
import pickle

def read_mnist(should_normalize=True):
    num_digits = 10
    data_size = 784
    X_train = np.zeros((trainSize*num_digits,data_size))
    X_val = np.zeros((valSize*num_digits,data_size))
    X_test = np.zeros((testSize*num_digits,data_size))
    Y_train = np.zeros((trainSize*num_digits,num_digits))
    Y_val = np.zeros((valSize*num_digits,num_digits))
    Y_test = np.zeros((testSize*num_digits,num_digits))
    for digit in range(num_digits):
        name = 'data/mnist_digit_' + str(digit) + '.csv'
        data_full = np.loadtxt(name)
        
        train = data_full[0:trainSize, :]
        val = data_full[trainSize:(trainSize + valSize), :]
        test = data_full[(trainSize + valSize):(trainSize + valSize + testSize), :]
        
        if should_normalize:
            train = normalize(train)
            val = normalize(val)
            test = normalize(test)

        X_train[digit*trainSize:(digit+1)*trainSize,:] = train
        Y_train[digit*trainSize:(digit+1)*trainSize,digit] = 1
        X_val[digit*valSize:(digit+1)*valSize,:] = val
        Y_val[digit*valSize:(digit+1)*valSize,digit] = 1
        X_test[digit*testSize:(digit+1)*testSize,:] = test
        Y_test[digit*testSize:(digit+1)*testSize,digit] = 1


    # randomly shuffle data so its not in order of digit
    rand_train_inds = np.arange(num_digits*trainSize)
    np.random.shuffle(rand_train_inds)
    X_train = X_train[rand_train_inds,:]
    Y_train = Y_train[rand_train_inds,:]

    rand_val_inds = np.arange(num_digits*valSize)
    np.random.shuffle(rand_val_inds)
    X_val = X_val[rand_val_inds,:]
    Y_val = Y_val[rand_val_inds,:]

    rand_test_inds = np.arange(num_digits*testSize)
    np.random.shuffle(rand_test_inds)
    X_test = X_test[rand_test_inds,:]
    Y_test = Y_test[rand_test_inds,:]

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def normalize(data):
    return (2 * data / 255 - np.ones(np.shape(data))) 

def write_to_pickle():
    X_train, X_val, X_test, Y_train, Y_val, Y_test = read_mnist()
    
    f = open('pickle_files/X_train.p', 'wb')
    pickle.dump(X_train,f)
    f.close()

    f = open('pickle_files/X_val.p', 'wb')
    pickle.dump(X_val,f)
    f.close()

    f = open('pickle_files/X_test.p', 'wb')
    pickle.dump(X_test,f)
    f.close()

    f = open('pickle_files/Y_train.p', 'wb')
    pickle.dump(Y_train,f)
    f.close()

    f = open('pickle_files/Y_val.p', 'wb')
    pickle.dump(Y_val,f)
    f.close()

    f = open('pickle_files/Y_test.p', 'wb')
    pickle.dump(Y_test,f)
    f.close()

 

def load_pickle():
    f = open('pickle_files/X_train.p', 'r')
    X_train = pickle.load(f)
    f.close()

    f = open('pickle_files/X_val.p', 'r')
    X_val = pickle.load(f)
    f.close()

    f = open('pickle_files/X_test.p', 'r')
    X_test = pickle.load(f)
    f.close()

    f = open('pickle_files/Y_train.p', 'r')
    Y_train = pickle.load(f)
    f.close()

    f = open('pickle_files/Y_val.p', 'r')
    Y_val = pickle.load(f)
    f.close()

    f = open('pickle_files/Y_test.p', 'r')
    Y_test = pickle.load(f)
    f.close()

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def plot():
    tr_accs = [42.049999999999997, 52.449999999999996, 57.649999999999999, 63.299999999999997, 66.399999999999991, 67.700000000000003, 70.900000000000006, 70.599999999999994, 72.0, 74.900000000000006, 76.5, 77.149999999999991, 80.699999999999989, 81.950000000000003, 81.400000000000006, 82.899999999999991, 83.650000000000006, 81.849999999999994, 84.0, 84.099999999999994, 85.0, 85.399999999999991, 86.299999999999997, 86.450000000000003, 88.349999999999994, 88.900000000000006, 87.599999999999994, 88.649999999999991, 90.149999999999991, 90.299999999999997, 91.649999999999991, 90.900000000000006, 91.75, 92.900000000000006, 93.650000000000006, 93.450000000000003, 93.700000000000003, 93.25, 93.25, 94.049999999999997, 91.5, 94.299999999999997, 94.0, 94.899999999999991, 94.299999999999997, 94.349999999999994, 94.699999999999989, 95.549999999999997, 95.899999999999991, 95.150000000000006, 96.150000000000006, 96.450000000000003, 95.799999999999997, 96.899999999999991, 96.850000000000009, 95.950000000000003, 96.150000000000006, 97.450000000000003, 97.0, 97.099999999999994, 97.549999999999997, 97.650000000000006, 97.850000000000009, 97.599999999999994, 97.799999999999997, 97.099999999999994, 97.350000000000009, 98.0, 98.200000000000003, 98.200000000000003, 97.899999999999991, 97.599999999999994, 97.950000000000003, 97.950000000000003, 98.400000000000006, 98.25, 98.299999999999997, 98.400000000000006, 98.450000000000003, 98.450000000000003, 98.299999999999997, 98.450000000000003, 98.550000000000011, 98.75, 98.5, 98.799999999999997, 98.799999999999997, 98.799999999999997, 98.799999999999997, 98.75, 98.799999999999997, 98.799999999999997, 99.0, 95.799999999999997, 97.899999999999991, 97.150000000000006, 97.799999999999997, 98.950000000000003, 98.700000000000003, 98.700000000000003]
    tr_accs = [42.049999999999997, 52.449999999999996, 57.649999999999999, 63.299999999999997, 66.399999999999991, 67.700000000000003, 70.900000000000006, 70.599999999999994, 72.0, 74.900000000000006, 76.5, 77.149999999999991, 80.699999999999989, 81.950000000000003, 81.400000000000006, 82.899999999999991, 83.650000000000006, 81.849999999999994, 84.0, 84.099999999999994, 85.0, 85.399999999999991, 86.299999999999997, 86.450000000000003, 88.349999999999994, 88.900000000000006, 87.599999999999994, 88.649999999999991, 90.149999999999991, 90.299999999999997, 91.649999999999991, 90.900000000000006, 91.75, 92.900000000000006, 93.650000000000006, 93.450000000000003, 93.700000000000003, 93.25, 93.25, 94.049999999999997, 91.5, 94.299999999999997, 94.0, 94.899999999999991, 94.299999999999997, 94.349999999999994, 94.699999999999989, 95.549999999999997, 95.899999999999991, 95.150000000000006, 96.150000000000006, 96.450000000000003, 95.799999999999997, 96.899999999999991, 96.850000000000009, 95.950000000000003, 96.150000000000006, 97.450000000000003, 97.0, 97.099999999999994, 97.549999999999997, 97.650000000000006, 97.850000000000009, 97.599999999999994, 97.799999999999997, 97.099999999999994, 97.350000000000009, 98.0, 98.200000000000003, 98.200000000000003, 97.899999999999991, 97.599999999999994, 97.950000000000003, 97.950000000000003, 98.400000000000006, 98.25, 98.299999999999997, 98.400000000000006, 98.450000000000003, 98.450000000000003, 98.299999999999997, 98.450000000000003, 98.550000000000011, 98.75, 98.5, 98.799999999999997, 98.799999999999997, 98.799999999999997, 98.799999999999997, 98.75, 98.799999999999997, 98.799999999999997, 99.0, 95.799999999999997, 97.899999999999991, 97.150000000000006, 97.799999999999997, 98.950000000000003, 98.700000000000003, 98.700000000000003]
    val_accs = [36.5, 46.799999999999997, 52.800000000000004, 57.899999999999999, 60.399999999999999, 61.299999999999997, 62.5, 63.399999999999999, 64.299999999999997, 66.299999999999997, 69.300000000000011, 70.5, 72.599999999999994, 74.200000000000003, 75.099999999999994, 74.0, 73.799999999999997, 72.299999999999997, 74.5, 75.5, 75.5, 75.700000000000003, 77.400000000000006, 77.5, 78.600000000000009, 79.800000000000011, 79.100000000000009, 79.600000000000009, 80.400000000000006, 80.299999999999997, 80.5, 81.099999999999994, 81.5, 82.600000000000009, 82.200000000000003, 82.400000000000006, 82.699999999999989, 80.299999999999997, 82.099999999999994, 82.600000000000009, 81.5, 82.600000000000009, 83.5, 83.299999999999997, 83.599999999999994, 83.299999999999997, 84.200000000000003, 84.099999999999994, 84.700000000000003, 84.399999999999991, 84.700000000000003, 84.799999999999997, 84.5, 84.799999999999997, 85.0, 84.299999999999997, 85.0, 85.700000000000003, 86.099999999999994, 85.900000000000006, 85.599999999999994, 85.900000000000006, 85.599999999999994, 86.200000000000003, 85.900000000000006, 85.299999999999997, 85.799999999999997, 86.099999999999994, 86.099999999999994, 86.299999999999997, 86.5, 86.599999999999994, 85.700000000000003, 86.099999999999994, 86.0, 86.0, 86.0, 85.799999999999997, 86.5, 86.0, 86.0, 86.599999999999994, 86.099999999999994, 86.5, 86.299999999999997, 86.599999999999994, 86.599999999999994, 86.5, 86.5, 86.5, 86.400000000000006, 86.599999999999994, 86.5, 84.5, 85.200000000000003, 84.599999999999994, 85.799999999999997, 86.299999999999997, 86.5, 86.5]
    epoch_list = range(len(tr_accs))
    pl.plot(epoch_list, tr_accs,'-o', label='Training')
    pl.plot(epoch_list, val_accs,'-x', label='Validation')

    pl.xlabel('Epoch')
    pl.ylabel('Accuracy (\%)')
    pl.legend(loc='lower right')
    pl.savefig('../paper/figures/1_5_acc')
    pl.show()

if __name__ == '__main__':
    font = {'family' : 'serif',
            'size'   : 20}
    pl.rc('font', **font)
    pl.rc('text', usetex=True)

    # plot()
    # assert(0)

    trainSize = 200
    valSize = 100
    testSize = 100
    num_digits = 10

    print "Loading MNIST dataset..."
    X_train, X_val, X_test, Y_train, Y_val, Y_test = read_mnist(should_normalize=True)
    # write_to_pickle()
    # X_train, X_val, X_test, Y_train, Y_val, Y_test = load_pickle()
    print "Loaded MNIST dataset."

    num_hidden_layers = 1
    hidden_layer_sizes = [100]
    input_size = 784
    output_size = 10
    network = nn.NeuralNetwork(num_hidden_layers, hidden_layer_sizes, input_size, output_size)
    learning_rate = 1e-6
    num_epochs = 50
    prev_acc = 100.0
    acc = 0.0
    num_epochs_used = 0
    num_epochs = 1
    loss = 0.0
    prev_loss = 100.0
    tr_accs = []
    val_accs = []
    while abs(loss - prev_loss) > 0.05 and num_epochs_used < 100:
        prev_loss = loss
        prev_acc = acc
        num_epochs_used += 1
        network.train(X_train, Y_train, learning_rate, num_epochs)

        # training accuracy
        num_pts = trainSize*num_digits
        preds, acc = network.predict(X_train, np.argmax(Y_train,axis=1))
        errors = np.sum(np.argmax(Y_train,axis=1) != preds)
        train_acc = 100*(1-errors/float(num_pts))

        loss = 0.0
        num_pts = valSize*num_digits
        preds, acc = network.predict(X_val, np.argmax(Y_val,axis=1))
        errors = np.sum(np.argmax(Y_val,axis=1) != preds)
        val_acc = 100*(1-errors/float(num_pts))
        for i in range(len(Y_val)):
            probs, pred = network.output(X_val[i,:])
            loss += network.compute_loss(probs, Y_val[i])
        print "Epoch %i: Train Acc: %.2f, Val Acc: %.2f, Loss: %.2f" %(num_epochs_used, train_acc, val_acc, loss)
        tr_accs.append(train_acc)
        val_accs.append(val_acc)

    print tr_accs
    print val_accs

    # Training Acc
    num_pts = trainSize*num_digits
    preds, acc = network.predict(X_train, np.argmax(Y_train,axis=1))
    errors = np.sum(np.argmax(Y_train,axis=1) != preds)
    train_acc = 100*(1-errors/float(num_pts))

    # Test Acc
    num_pts = testSize*num_digits
    preds, acc = network.predict(X_test, np.argmax(Y_test,axis=1))
    errors = np.sum(np.argmax(Y_test,axis=1) != preds)
    test_acc = 100*(1-errors/float(num_pts))
 
    print "Num Epochs: %i, Training Accuracy: %.2f, Test Acc: %.2f" %(num_epochs_used, train_acc, test_acc)



    # plotDecisionBoundary(X_test, np.asarray(Y_test, dtype = np.int32).reshape(-1), 
    #                      network.evaluate, [0])