import numpy as np
from plotBoundary import *
import pylab as pl
from cvxopt import matrix, solvers

# import your SVM training code

# parameters
name = '1'
print '======Training======'
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()

# X = np.array([np.array([2.,2.]), np.array([2.,3.]), \
#     np.array([0.,-1.]), np.array([-3.,-2.])])
# Y = np.array([np.array([1.]), np.array([1.]), \
#     np.array([-1.]), np.array([-1.])])
# print X, Y

C = 1.0

n = np.shape(X)[0]

P = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        P[i,j] = Y[i]*Y[j]*np.dot(X[i].T, X[j])
P = matrix(P)
q = matrix(-np.ones((n,1)))
G = matrix(np.vstack([-np.eye(n), np.eye(n)]))
h = matrix(np.vstack([np.zeros((n,1)), C*np.ones((n,1))]))
A = matrix(Y.T)
b = matrix(0.0)

print "P = ", P
print "q = ", q
print "G = ", G
print "h = ", h
print "A = ", A
print "b = ", b

solution = solvers.qp(P, q, G, h, A, b)
alpha = np.array(solution['x'])
nonzero_alpha_inds = np.where(abs(alpha) > 1e-4)[0]
zero_alpha_inds = np.setdiff1d(np.arange(n),nonzero_alpha_inds)
alpha[zero_alpha_inds] = 0.0
print alpha
w = np.sum(np.multiply(np.multiply(alpha, Y), X), axis=0)
print "w:", w

# From Eqn 7.37 in Bishop
bias = 0.0
for n in nonzero_alpha_inds:
    s = 0.0
    for m in np.arange(n):
        s += alpha[m]*Y[m]*np.dot(X[n].T, X[m])
    bias += Y[n] - s
bias /= np.shape(nonzero_alpha_inds)[0]

print "bias:", bias
# bias = -0.55


def predictSVM(x):
    return np.dot(w, x.T) + bias

# print predictSVM()


### TODO ###
# Define the predictSVM(x) function, which uses trained parameters
### TODO ###

# plot training results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')


# print '======Validation======'
# # load data from csv files
# validate = loadtxt('data/data'+name+'_validate.csv')
# X = validate[:, 0:2]
# Y = validate[:, 2:3]
# # plot validation results
# plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')
pl.show()
