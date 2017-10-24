from numpy import *
from plotBoundary import *
import pylab as pl
# import your LR training code

# load data from csv files
train = loadtxt('data/data3_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# Carry out training.
epochs = 1000;
lmbda = .02;
gamma = 2e-1;

### TODO: Compute the kernel matrix ###
def gaussian_kernel(x1, x2):
	return exp(-gamma*linalg.norm(x1 - x2, axis=1))

def gaussian_kernel_matrix(X):
	n = len(X)
	p = len(X[0])
	K = array([gaussian_kernel(x1.reshape(1, p), x2.reshape(1, p)) for x1 in X for x2 in X])
	K = K.reshape(n, n)
	return K

### TODO: Implement train_gaussianSVM ###
def train_gaussianSVM(X, Y, K, l, max_epochs = 1000):
	t = 0
	epoch = 0
	n = len(X)
	alpha = zeros(n)

	while epoch < max_epochs:
		for i in range(n):
			t += 1
			learning_rate = 1.0/(t*l)
			discriminant = Y[i] * dot(alpha, K[i, :])
			if discriminant < 1:
				alpha[i] = (1 - learning_rate*l)*alpha[i] + learning_rate*Y[i]
			else:
				alpha[i] = (1 - learning_rate*l)*alpha[i]

		epoch += 1
	return alpha

# Define the predict_gaussianSVM(x) function, which uses trained parameters, alpha
### TODO:  define predict_gaussianSVM(x) ###

def predict_gaussianSVM(x):
	# print "asdflkj: ", gaussian_kernel(X, x)
	return dot(alpha, gaussian_kernel(X, x))


for gamma in [2E-2, 2E-1, 2E0, 2E1, 2E2]:
	K = gaussian_kernel_matrix(X)
	alpha = train_gaussianSVM(X, Y, K, lmbda, epochs)

	# plot training results
	plotDecisionBoundary(X, Y, predict_gaussianSVM, [-1,0,1], title = 'Gaussian Kernel SVM, gamma = %E' % gamma)
	pl.show()