import numpy as np
from plotBoundary import *
import pylab as pl
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False

# import your SVM training code

# # parameters
# name = '1'
# print '======Training======'
# # load data from csv files
# train = loadtxt('data/data'+name+'_train.csv')
# # use deep copy here to make cvxopt happy
# X = train[:, 0:2].copy()
# Y = train[:, 2:3].copy()

X = np.array([np.array([2.,2.]), np.array([2.,3.]), \
    np.array([0.,-1.]), np.array([-3.,-2.])])
Y = np.array([np.array([1.]), np.array([1.]), \
    np.array([-1.]), np.array([-1.])])
# print X, Y

def trainSVM(X,Y,C=1.0):
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

	solution = solvers.qp(P, q, G, h, A, b)
	alpha = np.array(solution['x'])
	nonzero_alpha_inds = np.where(abs(alpha) > 1e-4)[0]
	zero_alpha_inds = np.setdiff1d(np.arange(n),nonzero_alpha_inds)
	alpha[zero_alpha_inds] = 0.0
	w = np.sum(np.multiply(np.multiply(alpha, Y), X), axis=0)

	# From Eqn 7.37 in Bishop
	bias = 0.0
	for n in nonzero_alpha_inds:
	    s = 0.0
	    for m in nonzero_alpha_inds:
	        s += alpha[m]*Y[m]*np.dot(X[n].T, X[m])
	    bias += Y[n] - s
	bias /= np.shape(nonzero_alpha_inds)[0]

	return w, bias


def predictSVM(x):
    return np.dot(w, x.T) + bias

#############
# Simple Dataset
##############
# w, bias = trainSVM(X, Y)
# print "w:", w
# print "bias:", bias
# plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1])
# pl.savefig("../paper/figures/2_1_decisions")



#############
# Decision boundaries for 4 datasets
###############

datasets = [1,2,3,4]
colors = ['g','b','y','r','m','c']
C = 1.0

pl.figure()
for dataset in datasets:
    train = loadtxt('data/data'+str(dataset)+'_train.csv')
    X_train = train[:,0:2]
    Y_train = train[:,2:3]
    validate = loadtxt('data/data'+str(dataset)+'_validate.csv')
    X_val = validate[:,0:2]
    Y_val = validate[:,2:3]
    # test = loadtxt('data/data'+str(dataset)+'_test.csv')
    # X_test = test[:,0:2]
    # Y_test = np.squeeze(test[:,2:3])
    num_training_pts = float(np.shape(Y_train)[0])
    num_val_pts = float(np.shape(Y_val)[0])

    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    h = max((x_max-x_min)/200., (y_max-y_min)/200.)
    xx, yy = meshgrid(arange(x_min, x_max, h),
                      arange(y_min, y_max, h))

    pl.subplot(2,4,dataset)
    pl.scatter(X_train[:, 0], X_train[:, 1], c=(1.-Y_train), s=50, cmap = pl.cm.cool)
    w, bias = trainSVM(X_train,Y_train,C=C)

    # Plot training data/boundaries
    zz = array([predictSVM(x) for x in c_[xx.ravel(), yy.ravel()]])
    zz = zz.reshape(xx.shape)
    CS = pl.contour(xx, yy, zz, [-1, 0, 1], linestyles = 'solid', linewidths = 2)
    pl.clabel(CS, fontsize=9, inline=1)
    pl.axis('tight')

    pl.subplot(2,4,dataset+4)
    pl.scatter(X_val[:, 0], X_val[:, 1], c=(1.-Y_val), s=50, cmap = pl.cm.cool)
    zz = array([predictSVM(x) for x in c_[xx.ravel(), yy.ravel()]])
    zz = zz.reshape(xx.shape)
    CS = pl.contour(xx, yy, zz, [-1, 0, 1], linestyles = 'solid', linewidths = 2)
    pl.clabel(CS, fontsize=9, inline=1)
    pl.axis('tight')

    preds = predictSVM(X_train)
    preds[preds > 0] = 1
    preds[preds <= 0] = -1
    errors = np.sum(np.squeeze(Y_train) != preds)
    train_accuracy = 100*(1-errors/num_training_pts)

    preds = predictSVM(X_val)
    preds[preds > 0] = 1
    preds[preds <= 0] = -1
    errors = np.sum(np.squeeze(Y_val) != preds)
    val_accuracy = 100*(1-errors/num_val_pts)
    print "Dataset %i: Training Set Accuracy: %.3f, Validation Set Accuracy: %.3f" %(dataset, train_accuracy, val_accuracy)
pl.savefig("../paper/figures/2_2_decisions")


pl.show()





# print '======Validation======'
# # load data from csv files
# validate = loadtxt('data/data'+name+'_validate.csv')
# X = validate[:, 0:2]
# Y = validate[:, 2:3]
# # plot validation results
# plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')
# pl.show()
