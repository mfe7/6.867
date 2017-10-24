import numpy as np
from plotBoundary import *
import pylab as pl
import lr_train

# parameters
name = '1'
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
X_train = train[:,0:2]
Y_train = train[:,2:3]
validate = loadtxt('data/data'+name+'_validate.csv')
X_val = validate[:,0:2]
Y_val = validate[:,2:3]
test = loadtxt('data/data'+name+'_test.csv')
X_test = test[:,0:2]
Y_test = test[:,2:3]

#################################

# def lr_gradient(x, y, w, w0, l2=True):
#     # returns gradient of w, w0
#     u = dot(w, x) + w0
#     if l2:
#         return (-y*x) * exp(-y*u)/(1 + exp(-y*u)) + 2*lmbda*w, -y*exp(-y*u)/(1 + exp(-y*u))
#     else:
#         return (-y*x) * exp(-y*u)/(1 + exp(-y*u)) + lmbda, -y*exp(-y*u)/(1 + exp(-y*u))

# def lr_objective(x, y, w, w0):
#     u = dot(x, w) + w0
#     return sum(log(1 + exp(-y*u)))

# def lr_train(X, Y):
#     last = inf
#     epoch = 0
#     N = len(X)
#     p = len(X[0])
    
#     learning_rate = .01

#     w = zeros(p)
#     w0 = 0


#     while epoch < max_epochs:
#         for i in range(N):
#             dw, dw0 = lr_gradient(X[i], Y[i], w, w0)
#             w -= learning_rate * dw
#             w0 -= learning_rate * dw0
#         epoch += 1

#     return w, w0

# def sigmoid(x):
#     return 1./(1 + np.exp(-x))

# def predictLR(x):
#     return sigmoid(dot(w, x) + w0)

#################################

# ########
# # LR with SGD
# #########
# pl.figure()
# max_epochs = 100
# x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
# y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
# h = max((x_max-x_min)/200., (y_max-y_min)/200.)
# xx, yy = meshgrid(arange(x_min, x_max, h),
#                   arange(y_min, y_max, h))

# for lmbda in [0,1]:
# # for lmbda in [0]:
#     w, w0 = lr_train(X_train, Y_train)
#     print w,w0
#     zz = array([predictLR(x) for x in c_[xx.ravel(), yy.ravel()]])
#     zz = zz.reshape(xx.shape)
#     CS = pl.contour(xx, yy, zz, [0.5], colors = 'green', linestyles = 'solid', linewidths = 2)
#     pl.clabel(CS, fontsize=9, inline=1, fmt="$\lambda="+str(lmbda)+"$")
# pl.scatter(X_train[:, 0], X_train[:, 1], c=(1.-Y_train), s=50, cmap = pl.cm.cool)
# pl.axis('tight')
# # pl.savefig("../paper/figures/1_1_decision")
# pl.show()


###################


# Carry out training.
# lr_coefs, lr_intercept = lr_train.sgd_train(X,Y,penalty='l1')


# Define the predictLR(x) function, which uses trained parameters
def predictLR(x):
    return np.dot(lr_coefs, x.T) + lr_intercept

# lambdas = [0.01, 0.1, 1, 10, 100, 1000]
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# h = max((x_max-x_min)/200., (y_max-y_min)/200.)
# xx, yy = meshgrid(arange(x_min, x_max, h),
#                   arange(y_min, y_max, h))
# values = [0.5]
# pl.figure()
# pl.subplot(1,2,1)
# for lamb in lambdas:
#     lr_coefs, lr_intercept = lr_train.lr_train(X,Y,penalty='l1', lamb=lamb)
#     zz = array([predictLR(x) for x in c_[xx.ravel(), yy.ravel()]])
#     zz = zz.reshape(xx.shape)
#     CS = pl.contour(xx, yy, zz, [0.5], colors = 'green', linestyles = 'solid', linewidths = 2)
#     pl.clabel(CS, fontsize=9, inline=1, fmt="$\lambda="+str(lamb)+"$")
# pl.scatter(X[:, 0], X[:, 1], c=(1.-Y), s=50, cmap = pl.cm.cool)
# pl.axis('tight')

# pl.subplot(1,2,2)
# for lamb in lambdas:
#     lr_coefs, lr_intercept = lr_train.lr_train(X,Y,penalty='l2', lamb=lamb)
#     zz = array([predictLR(x) for x in c_[xx.ravel(), yy.ravel()]])
#     zz = zz.reshape(xx.shape)
#     CS = pl.contour(xx, yy, zz, [0.5], colors = 'green', linestyles = 'solid', linewidths = 2)
#     pl.clabel(CS, fontsize=9, inline=1, fmt="$\lambda="+str(lamb)+"$")
# pl.scatter(X[:, 0], X[:, 1], c=(1.-Y), s=50, cmap = pl.cm.cool)
# pl.axis('tight')

# pl.savefig("../paper/figures/1_2_decision")
# pl.show()

#####
# Decision Boundaries
#####

# lambdas = np.logspace(-2,3,num=6)
# datasets = [1,2,3,4]
# colors = ['g','b','y','r','m','c']

# pl.figure()
# for dataset in datasets:
#     train = loadtxt('data/data'+str(dataset)+'_train.csv')
#     X_train = train[:,0:2]
#     Y_train = train[:,2:3]
#     validate = loadtxt('data/data'+str(dataset)+'_validate.csv')
#     X_val = validate[:,0:2]
#     Y_val = validate[:,2:3]
#     test = loadtxt('data/data'+str(dataset)+'_test.csv')
#     X_test = test[:,0:2]
#     Y_test = np.squeeze(test[:,2:3])
#     num_pts = float(np.shape(Y_test)[0])

#     x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
#     y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
#     h = max((x_max-x_min)/200., (y_max-y_min)/200.)
#     xx, yy = meshgrid(arange(x_min, x_max, h),
#                       arange(y_min, y_max, h))

#     pl.subplot(2,4,dataset)
#     pl.scatter(X_train[:, 0], X_train[:, 1], c=(1.-Y_train), s=50, cmap = pl.cm.cool)
#     # pl.scatter(X_test[:, 0], X_test[:, 1], c=(1.-Y_test), s=50, cmap = pl.cm.cool)

#     for i, lamb in enumerate(lambdas):
#         lr_coefs, lr_intercept = lr_train.lr_train(X_train,Y_train,penalty='l1', lamb=lamb)
#         values = [0.5]
#         zz = array([predictLR(x) for x in c_[xx.ravel(), yy.ravel()]])
#         zz = zz.reshape(xx.shape)
#         CS = pl.contour(xx, yy, zz, [0.5], colors = colors[i], linestyles = 'solid', linewidths = 2)
#         # pl.clabel(CS, fontsize=9, inline=1, fmt="$\lambda="+str(lamb)+"$")
#     pl.axis('tight')

#     pl.subplot(2,4,dataset+4)
#     pl.scatter(X_test[:, 0], X_test[:, 1], c=(1.-Y_test), s=50, cmap = pl.cm.cool)
#     for i, lamb in enumerate(lambdas):
#         lr_coefs, lr_intercept = lr_train.lr_train(X_train,Y_train,penalty='l2', lamb=lamb)
#         values = [0.5]
#         zz = array([predictLR(x) for x in c_[xx.ravel(), yy.ravel()]])
#         zz = zz.reshape(xx.shape)
#         CS = pl.contour(xx, yy, zz, [0.5], colors = colors[i], linestyles = 'solid', linewidths = 2)
#         # pl.clabel(CS, fontsize=9, inline=1, fmt="$\lambda="+str(lamb)+"$")


# pl.savefig("../paper/figures/1_2_decisions")
# pl.show()

#####
# Weights
#####

# lambdas = np.logspace(-2,3,num=6)
# datasets = [1,2,3,4]
# colors = ['g','b','y','r','m','c']

# pl.figure()
# for dataset in datasets:
#     train = loadtxt('data/data'+str(dataset)+'_train.csv')
#     X_train = train[:,0:2]
#     Y_train = train[:,2:3]


#     pl.subplot(2,4,dataset)
#     pl.xticks([0,1,2])

#     for i, lamb in enumerate(lambdas):
#         lr_coefs, lr_intercept = lr_train.lr_train(X_train,Y_train,penalty='l1', lamb=lamb)
#         n = np.shape(lr_coefs)[1]
#         coefs = pl.plot(range(n+1), np.hstack([lr_intercept, lr_coefs[0]]),'-o',label=r'$\lambda='+str(lamb)+'$')
#         if dataset==1:
#             pl.legend(loc='best')
#             pl.xlabel('Weight Index ($w_0$ is intercept)')
#             pl.ylabel('Weight Magnitude')




#     pl.subplot(2,4,dataset+4)
#     for i, lamb in enumerate(lambdas):
#         lr_coefs, lr_intercept = lr_train.lr_train(X_train,Y_train,penalty='l2', lamb=lamb)
#         coefs = pl.plot(range(n+1), np.hstack([lr_intercept, lr_coefs[0]]),'-o',label=r'$\lambda='+str(lamb)+'$')
#         # pl.legend(loc='upper center')

# pl.savefig("../paper/figures/1_2_all_weights")
# pl.show()

# lambdas = [0.01, 0.1, 1, 10, 100, 1000]
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# h = max((x_max-x_min)/200., (y_max-y_min)/200.)
# xx, yy = meshgrid(arange(x_min, x_max, h),
#                   arange(y_min, y_max, h))
# values = [0.5]
# pl.figure()
# pl.subplot(1,2,1)
# for lamb in lambdas:
#     lr_coefs, lr_intercept = lr_train.lr_train(X,Y,penalty='l1', lamb=lamb)
#     print "lamb:", lamb, lr_coefs
#     n = np.shape(lr_coefs)[1]
#     print n
#     print lr_intercept
#     print lr_coefs[0]
#     print np.hstack([lr_intercept, lr_coefs[0]])
#     coefs = pl.plot(range(n+1), np.hstack([lr_intercept, lr_coefs[0]]),'-o',label=r'$\lambda='+str(lamb)+'$')
# pl.legend(loc='upper center')
# pl.xticks([0, 1, 2])
# pl.ylim(-1, 10)
# pl.xlabel('Weight Index ($w_0$ is intercept)')
# pl.ylabel('Weight Magnitude')

# pl.subplot(1,2,2)
# for lamb in lambdas:
#     lr_coefs, lr_intercept = lr_train.lr_train(X,Y,penalty='l2', lamb=lamb)
#     n = np.shape(lr_coefs)[1]
#     coefs = pl.plot(range(n+1), np.hstack([lr_intercept, lr_coefs[0]]),'-o',label=r'$\lambda='+str(lamb)+'$')
# pl.legend(loc='upper center')
# pl.xticks([0, 1, 2])
# pl.ylim(-1, 10)
# pl.xlabel('Weight Index ($w_0$ is intercept)')
# pl.ylabel('Weight Magnitude')

# pl.savefig("../paper/figures/1_2_weights")
# pl.show()


######
# Accuracy Plot
#####
# lambdas = np.logspace(-2,6,num=50)
# datasets = [1,2,3,4]
# pl.figure()
# for dataset in datasets:
#     train = loadtxt('data/data'+str(dataset)+'_train.csv')
#     X_train = train[:,0:2]
#     Y_train = train[:,2:3]
#     validate = loadtxt('data/data'+str(dataset)+'_validate.csv')
#     X_val = validate[:,0:2]
#     Y_val = validate[:,2:3]
#     test = loadtxt('data/data'+str(dataset)+'_test.csv')
#     X_test = test[:,0:2]
#     Y_test = np.squeeze(test[:,2:3])
#     num_pts = float(np.shape(Y_test)[0])

#     pl.subplot(2,2,dataset)

#     l1_accuracy = []
#     for lamb in lambdas:
#         lr_coefs, lr_intercept = lr_train.lr_train(X_train,Y_train,penalty='l1', lamb=lamb)
#         preds = predictLR(X_test)
#         preds = np.squeeze(preds)
#         preds[preds > 0] = 1
#         preds[preds <= 0] = -1
#         errors = np.sum(Y_test != preds)
#         accuracy = 100*(1-errors/num_pts)
#         l1_accuracy.append(accuracy)

#     l2_accuracy = []
#     for lamb in lambdas:
#         lr_coefs, lr_intercept = lr_train.lr_train(X_train,Y_train,penalty='l2', lamb=lamb)
#         preds = predictLR(X_test)
#         preds = np.squeeze(preds)
#         preds[preds > 0] = 1
#         preds[preds <= 0] = -1
#         errors = np.sum(Y_test != preds)
#         accuracy = 100*(1-errors/num_pts)
#         l2_accuracy.append(accuracy)
#         # print "\nlamb=", lamb
#         # print "errors: %i out of %i (%d%% accuracy)"%(errors, num_pts, accuracy )

#     pl.semilogx(lambdas, l1_accuracy, label=r"$L_1$ reg.")
#     pl.semilogx(lambdas, l2_accuracy, label=r"$L_2$ reg.")
#     pl.xlabel("$\lambda$")
#     pl.ylabel("Test Set Accuracy (%)")
#     pl.ylim(0, 105)
#     pl.legend(loc="lower left")
# pl.savefig("../paper/figures/1_2_accuracy")
# pl.show()

# pl.subplot(1,2,2)
# for lamb in lambdas:
#     lr_coefs, lr_intercept = lr_train.lr_train(X,Y,penalty='l2', lamb=lamb)
#     zz = array([predictLR(x) for x in c_[xx.ravel(), yy.ravel()]])
#     zz = zz.reshape(xx.shape)
#     CS = pl.contour(xx, yy, zz, [0.5], colors = 'green', linestyles = 'solid', linewidths = 2)
#     pl.clabel(CS, fontsize=9, inline=1, fmt="$\lambda="+str(lamb)+"$")
# pl.scatter(X[:, 0], X[:, 1], c=(1.-Y), s=50, cmap = pl.cm.cool)
# pl.axis('tight')

# pl.savefig("../paper/figures/1_2_decision")
# pl.show()


#####
# Pick best params (1.3)
#####

# datasets = [1,2,3,4]
# lambdas = np.logspace(-2,3,num=21)

# for dataset in datasets:
#     train = loadtxt('data/data'+str(dataset)+'_train.csv')
#     X_train = train[:,0:2]
#     Y_train = train[:,2:3]
#     validate = loadtxt('data/data'+str(dataset)+'_validate.csv')
#     X_val = validate[:,0:2]
#     Y_val = np.squeeze(validate[:,2:3])
#     test = loadtxt('data/data'+str(dataset)+'_test.csv')
#     X_test = test[:,0:2]
#     Y_test = np.squeeze(test[:,2:3])

#     l1_accuracy = []
#     l2_accuracy = []

#     for i, lamb in enumerate(lambdas):
#         lr_coefs, lr_intercept = lr_train.lr_train(X_train,Y_train,penalty='l1', lamb=lamb)
#         preds = predictLR(X_val)
#         preds = np.squeeze(preds)
#         num_pts = float(np.shape(Y_val)[0])
#         preds[preds > 0] = 1
#         preds[preds <= 0] = -1
#         errors = np.sum(Y_val != preds)
#         accuracy = 100*(1-errors/num_pts)
#         l1_accuracy.append(accuracy)

#         lr_coefs, lr_intercept = lr_train.lr_train(X_train,Y_train,penalty='l2', lamb=lamb)
#         preds = predictLR(X_val)
#         preds = np.squeeze(preds)
#         num_pts = float(np.shape(Y_val)[0])
#         preds[preds > 0] = 1
#         preds[preds <= 0] = -1
#         errors = np.sum(Y_val != preds)
#         accuracy = 100*(1-errors/num_pts)
#         l2_accuracy.append(accuracy)

#     print "\nDataset", dataset
#     print "L1:", l1_accuracy, "max:", np.max(l1_accuracy), "lambda:", lambdas[np.argmax(l1_accuracy)]
#     print "L2:", l2_accuracy, "max:", np.max(l2_accuracy), "lambda:", lambdas[np.argmax(l2_accuracy)]
# print lambdas

####
# evaluate on test set
#####

lambdas = [0.01,0.01,1.78,1000]
penalties = ['l2','l2','l1','l2']
datasets = [1,2,3,4]
for dataset in datasets:
    train = loadtxt('data/data'+str(dataset)+'_train.csv')
    X_train = train[:,0:2]
    Y_train = np.squeeze(train[:,2:3])
    validate = loadtxt('data/data'+str(dataset)+'_validate.csv')
    X_val = validate[:,0:2]
    Y_val = np.squeeze(validate[:,2:3])
    test = loadtxt('data/data'+str(dataset)+'_test.csv')
    X_test = test[:,0:2]
    Y_test = np.squeeze(test[:,2:3])

    lr_coefs, lr_intercept = lr_train.lr_train(X_train,Y_train,penalty=penalties[dataset-1], lamb=lambdas[dataset-1])

    print "\nTest Dataset", dataset
    print "Lambda:", lambdas[dataset-1]
    print "Regularizer:", penalties[dataset-1]

    preds = predictLR(X_train)
    preds = np.squeeze(preds)
    num_pts = float(np.shape(Y_train)[0])
    preds[preds > 0] = 1
    preds[preds <= 0] = -1
    errors = np.sum(Y_train != preds)
    accuracy = 100*(1-errors/num_pts)
    print "Train Accuracy:", accuracy

    preds = predictLR(X_test)
    preds = np.squeeze(preds)
    num_pts = float(np.shape(Y_test)[0])
    preds[preds > 0] = 1
    preds[preds <= 0] = -1
    errors = np.sum(Y_test != preds)
    accuracy = 100*(1-errors/num_pts)
    print "Test Accuracy:", accuracy





