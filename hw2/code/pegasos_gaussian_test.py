import numpy as np
from plotBoundary import *
import pylab as pl
# import your LR training code

# # load data from csv files
# train = loadtxt('data/data3_train.csv')
# X = train[:,0:2]
# Y = train[:,2:3]

# Carry out training.
epochs = 100;
lmbda = .02;

def gaussian_rbf(x, x_prime, gamma):
    return np.exp(-gamma*np.linalg.norm(x - x_prime, axis=1))

def gaussian_rbf_matrix(x, gamma):
    n = len(x)
    p = len(x[0])
    K = np.zeros((n,n))
    for i in np.arange(n):
        for j in np.arange(n):
            K[i,j] = gaussian_rbf(x[i,:].reshape(1, p), x[j,:].reshape(1, p), gamma)
    return K

def train_gaussianSVM(X, Y, K, lmbda, epochs):
    N = np.shape(X)[0]
    d = np.shape(X)[1]
    inds = np.arange(N)
    t = 0
    alpha = np.zeros((N))
    epoch = 0
    while epoch < epochs:
        np.random.shuffle(inds)
        for i in inds:
            t += 1
            eta = 1.0/(t*lmbda)
            discrim = np.dot(alpha, K[i, :])
            if Y[i]*discrim < 1:
                alpha[i] = (1-eta*lmbda)*alpha[i] + eta*Y[i]
            else:
                alpha[i] = (1-eta*lmbda)*alpha[i]
        epoch += 1
    return alpha

def predict_gaussianSVM(x, alpha, x_train, gamma):
    return np.dot(alpha, gaussian_rbf(x_train, x, gamma))




#############
# Decision boundaries for 4 datasets
###############

datasets = [3]
# datasets = [1,2,3,4]
colors = ['g','b','y','r','m','c']
C = 1.0

# gammas = [2e2, 2e1, 2e0, 2e-1, 2e-2]
gammas = [2e2, 2e1, 2e0, 2e-1]

dataset = 3
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

pl.figure()
for i, gamma in enumerate(gammas):
    pl.subplot(1,4,i+1)
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    # h = max((x_max-x_min)/50., (y_max-y_min)/50.)
    h = max((x_max-x_min)/200., (y_max-y_min)/200.)
    xx, yy = meshgrid(arange(x_min, x_max, h),
                      arange(y_min, y_max, h))

    pl.scatter(X_train[:, 0], X_train[:, 1], c=(1.-Y_train), s=50, cmap = pl.cm.cool)

    K = gaussian_rbf_matrix(X_train, gamma)
    alpha = train_gaussianSVM(X_train,Y_train, K, lmbda, epochs)
    nonzero_alpha_inds = np.where(alpha > 1e-4)[0]
    print "gamma:", gamma, "num non-zero alphas:", len(nonzero_alpha_inds)
    # zz = array([predict_gaussianSVM(x, alpha, X_train, gamma) for x in c_[xx.ravel(), yy.ravel()]])
    # print zz
    # zz = zz.reshape(xx.shape)
    # CS = pl.contour(xx, yy, zz, [-1, 0, 1], linestyles = 'solid', linewidths = 2)
    # pl.clabel(CS, fontsize=9, inline=1)
    # pl.axis('tight')

# pl.savefig("../paper/figures/3_3_decisions")


pl.show()
