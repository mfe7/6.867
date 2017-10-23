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
    w = np.zeros((np.shape(X)[1]))
    w0 = 0
    epoch = 0
    while epoch < max_epochs:
        # np.random.shuffle(inds)
        for i in range(n):
            j = inds[i]
            t += 1
            eta = 1.0/(t*lamb)
            if Y[j]*(np.dot(w.T,X[j,:])+w0) < 1:
                e_g = Y[i]*X[i,:]
                w = (1-eta*lamb)*w+eta*e_g
                w0 += eta*Y[i]
            else:
                w = (1-eta*lamb)*w
        epoch += 1

    return w, w0

def predict_linearSVM(x, w, bias):
    return np.dot(w, x.T) + bias

max_epochs = 100
# lamb = 2e0
# w, bias = train_pegasos(X, Y, lamb, max_epochs)
# print w

# Define the predict_linearSVM(x) function, which uses global trained parameters, w
### TODO: define predict_linearSVM(x) ###

# plot training results
# plotDecisionBoundary(X, Y, predict_linearSVM, [-1,0,1], title = 'Linear SVM')
# pl.show()

lambs = [2e0, 2e-1, 2e-2, 2e-4]
for i, lamb in enumerate(lambs):
    # w, bias = lin_pegasos(X, Y, lamb, max_epochs)
    w, bias = train_pegasos(X, Y, lamb, max_epochs)
    print w, bias

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = max((x_max-x_min)/200., (y_max-y_min)/200.)
    xx, yy = meshgrid(arange(x_min, x_max, h),
                      arange(y_min, y_max, h))

    pl.subplot(1,4,i+1)
    pl.scatter(X[:, 0], X[:, 1], c=(1.-Y), s=50, cmap = pl.cm.cool)

    # Plot training data/boundaries
    zz = array([predict_linearSVM(x, w, bias) for x in c_[xx.ravel(), yy.ravel()]])
    zz = zz.reshape(xx.shape)
    CS = pl.contour(xx, yy, zz, [-1, 0, 1], linestyles = 'solid', linewidths = 2)
    pl.clabel(CS, fontsize=9, inline=1)
    pl.axis('tight')
pl.savefig("../paper/figures/3_2_lambdas")
pl.show()
