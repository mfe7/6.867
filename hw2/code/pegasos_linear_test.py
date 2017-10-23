import numpy as np
from plotBoundary import *
import pylab as pl
# import your LR training code


# load data from csv files
train = loadtxt('data/data3_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# Carry out training.
### TODO %%%

def train_pegasos(X, Y, lamb, max_epochs):
    n = np.shape(X)[0]
    inds = np.arange(n)
    t = 0
    w = [np.array([0,0])]
    epoch = 0
    while epoch < max_epochs:
        np.random.shuffle(inds)
        for i in range(n):
            j = inds[i]
            t += 1
            eta = 1.0/(t*lamb)
            if Y[j]*np.dot(w[-1].T,X[j,:]) < 1:
                w.append((1-eta*lamb)*w[-1]+eta*Y[i]*X[i,:])
            else:
                w.append((1-eta*lamb)*w[-1])
        epoch += 1

    # # From Eqn 7.37 in Bishop
    # bias = 0.0
    # for n in nonzero_alpha_inds:
    #     s = 0.0
    #     for m in nonzero_alpha_inds:
    #         s += alpha[m]*Y[m]*np.dot(X[n].T, X[m])
    #     bias += Y[n] - s
    # bias /= np.shape(nonzero_alpha_inds)[0]
    bias = 0.0
    return w[-1], bias

def predict_linearSVM(x):
    return np.dot(w, x.T) + bias

max_epochs = 20
lamb = 2e-5
w, bias = train_pegasos(X, Y, lamb, max_epochs)
print w

# Define the predict_linearSVM(x) function, which uses global trained parameters, w
### TODO: define predict_linearSVM(x) ###

# plot training results
plotDecisionBoundary(X, Y, predict_linearSVM, [-1,0,1], title = 'Linear SVM')
pl.show()

