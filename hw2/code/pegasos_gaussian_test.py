import numpy as np
from plotBoundary import *
import pylab as pl
# import your LR training code

# load data from csv files
train = loadtxt('data/data3_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# Carry out training.
epochs = 300;
lmbda = .02;
gamma = 2e-2;

def gaussian_kernel(X):
    n = np.shape(X)[0]
    K = zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j] = -gamma*np.linalg.norm(X[i,:] - X[j,:])**2
    return K




def train_gaussianSVM(X, Y, K, lmbda, epochs):
    N = np.shape(X)[0]
    d = np.shape(X)[1]
    inds = np.arange(N)
    t = 0
    alpha = np.zeros((N))
    epoch = 0
    while epoch < epochs:
        print epoch
        np.random.shuffle(inds)
        for i in inds:
            t += 1
            eta = 1.0/(t*lmbda)
            discrim = 0
            for j in range(N):
                discrim += alpha[j]*K[j,i]
            if Y[i]*discrim < 1:
                alpha[i] = (1-eta*lmbda)*alpha[i] + eta*Y[i]
            else:
                alpha[i] = (1-eta*lmbda)*alpha[i]
        epoch += 1
    return alpha


### TODO: Implement train_gaussianSVM ###
# K = gaussian_kernel(X)
# alpha = train_gaussianSVM(X, Y, K, lmbda, epochs);

def predict_gaussianSVM(x, alpha, x_train, y_train):
    output = 0
    for i in np.arange(np.shape(alpha)[0]):
        k = -gamma*np.linalg.norm(x_train[i,:] - x)**2
        output += alpha[i]*y_train[i]*k
    return output



#############
# Decision boundaries for 4 datasets
###############

datasets = [3]
# datasets = [1,2,3,4]
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
    h = max((x_max-x_min)/50., (y_max-y_min)/50.)
    # h = max((x_max-x_min)/200., (y_max-y_min)/200.)
    xx, yy = meshgrid(arange(x_min, x_max, h),
                      arange(y_min, y_max, h))

    pl.scatter(X_train[:, 0], X_train[:, 1], c=(1.-Y_train), s=50, cmap = pl.cm.cool)
    K = gaussian_kernel(X)

    print "Training gaussian SVM on dataset:", dataset
    alpha = train_gaussianSVM(X_train,Y_train, K, lmbda, epochs)
    print "Done training."
    # Plot training data/boundaries
    print "Predicting."
    zz = array([predict_gaussianSVM(x, alpha, X_train, Y_train) for x in c_[xx.ravel(), yy.ravel()]])
    print "done predicting"
    zz = zz.reshape(xx.shape)
    CS = pl.contour(xx, yy, zz, [-1, 0, 1], linestyles = 'solid', linewidths = 2)
    pl.clabel(CS, fontsize=9, inline=1)
    pl.axis('tight')

pl.savefig("../paper/figures/3_3_decisions")


pl.show()


# Define the predict_gaussianSVM(x) function, which uses trained parameters, alpha
### TODO:  define predict_gaussianSVM(x) ###

# plot training results
# plotDecisionBoundary(X, Y, predict_gaussianSVM, [-1,0,1], title = 'Gaussian Kernel SVM')
# pl.show()
