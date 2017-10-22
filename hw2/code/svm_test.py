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

gaussianRBF_lambda = 1.0

def linearKernel(x1, x2):
    return np.dot(x1.T, x2)

def gaussianRBFKernel(x1, x2):
    return np.exp(-gaussianRBF_lambda*np.linalg.norm(x1-x2)**2)

def trainSVM(X,Y,kernel=linearKernel, C=1.0):
	n = np.shape(X)[0]

	P = np.zeros((n,n))
	for i in range(n):
	    for j in range(n):
	        P[i,j] = Y[i]*Y[j]*kernel(X[i], X[j])
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

	return w, bias, alpha, nonzero_alpha_inds


# def predictSVM(x):
#     return np.dot(w, x.T) + bias

def predictSVM(x, x_train, y_train, bias, alpha, kernel=linearKernel):
    output = 0
    for i in nonzero_alpha_inds:
        k = kernel(x_train[i,:],x)
        output += alpha[i]*y_train[i]*k
    output += bias
    return output

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

# datasets = [1,2,3,4]
# colors = ['g','b','y','r','m','c']
# C = 1.0

# pl.figure()
# for dataset in datasets:
#     train = loadtxt('data/data'+str(dataset)+'_train.csv')
#     X_train = train[:,0:2]
#     Y_train = train[:,2:3]
#     validate = loadtxt('data/data'+str(dataset)+'_validate.csv')
#     X_val = validate[:,0:2]
#     Y_val = validate[:,2:3]
#     # test = loadtxt('data/data'+str(dataset)+'_test.csv')
#     # X_test = test[:,0:2]
#     # Y_test = np.squeeze(test[:,2:3])
#     num_training_pts = float(np.shape(Y_train)[0])
#     num_val_pts = float(np.shape(Y_val)[0])

#     x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
#     y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
#     h = max((x_max-x_min)/200., (y_max-y_min)/200.)
#     xx, yy = meshgrid(arange(x_min, x_max, h),
#                       arange(y_min, y_max, h))

#     pl.subplot(2,4,dataset)
#     pl.scatter(X_train[:, 0], X_train[:, 1], c=(1.-Y_train), s=50, cmap = pl.cm.cool)
#     w, bias = trainSVM(X_train,Y_train,C=C)

#     # Plot training data/boundaries
#     zz = array([predictSVM(x) for x in c_[xx.ravel(), yy.ravel()]])
#     zz = zz.reshape(xx.shape)
#     CS = pl.contour(xx, yy, zz, [-1, 0, 1], linestyles = 'solid', linewidths = 2)
#     pl.clabel(CS, fontsize=9, inline=1)
#     pl.axis('tight')

#     pl.subplot(2,4,dataset+4)
#     pl.scatter(X_val[:, 0], X_val[:, 1], c=(1.-Y_val), s=50, cmap = pl.cm.cool)
#     zz = array([predictSVM(x) for x in c_[xx.ravel(), yy.ravel()]])
#     zz = zz.reshape(xx.shape)
#     CS = pl.contour(xx, yy, zz, [-1, 0, 1], linestyles = 'solid', linewidths = 2)
#     pl.clabel(CS, fontsize=9, inline=1)
#     pl.axis('tight')

#     preds = predictSVM(X_train)
#     preds[preds > 0] = 1
#     preds[preds <= 0] = -1
#     errors = np.sum(np.squeeze(Y_train) != preds)
#     train_accuracy = 100*(1-errors/num_training_pts)

#     preds = predictSVM(X_val)
#     preds[preds > 0] = 1
#     preds[preds <= 0] = -1
#     errors = np.sum(np.squeeze(Y_val) != preds)
#     val_accuracy = 100*(1-errors/num_val_pts)
#     print "Dataset %i: Training Set Accuracy: %.3f, Validation Set Accuracy: %.3f" %(dataset, train_accuracy, val_accuracy)
# pl.savefig("../paper/figures/2_2_decisions")


# pl.show()


#############
# Effect of C
##############


# datasets = [2]
datasets = [1,2,3,4]
colors = ['g','b','y','r','m','c']
C = 1.0
Cs = [0.01, 0.1, 1, 10, 100]

Cs = [0.01, 0.01, 1.0, 1.0]
lmbds = [0.1, 0.1, 1.0, 1.0]

margins_lin = []
sv_lin = []
margins_rbf = []
sv_rbf = []

pl.figure()
for dataset in datasets:
    train = loadtxt('data/data'+str(dataset)+'_train.csv')
    X_train = train[:,0:2]
    Y_train = train[:,2:3]
    validate = loadtxt('data/data'+str(dataset)+'_validate.csv')
    X_val = validate[:,0:2]
    Y_val = validate[:,2:3]
    num_training_pts = float(np.shape(Y_train)[0])
    num_val_pts = float(np.shape(Y_val)[0])

    x_min, x_max = X_val[:, 0].min() - 1, X_val[:, 0].max() + 1
    y_min, y_max = X_val[:, 1].min() - 1, X_val[:, 1].max() + 1
    h = max((x_max-x_min)/200., (y_max-y_min)/200.)
    xx, yy = meshgrid(arange(x_min, x_max, h),
                      arange(y_min, y_max, h))

    kernel = linearKernel
    w, bias, alpha, nonzero_alpha_inds = trainSVM(X_train,Y_train,kernel=kernel,C=1.0)
    pl.subplot(2,4,dataset)
    pl.scatter(X_train[:, 0], X_train[:, 1], c=(1.-Y_train), s=50, cmap = pl.cm.cool)

    # print "Dataset:", dataset
    # mlin = []
    # slin = []
    # mrbf = []
    # srbf = []
    # for C in Cs:
    #     print "C:", C
    #     kernel = linearKernel
    #     w, bias, alpha, nonzero_alpha_inds = trainSVM(X_train,Y_train,kernel=kernel,C=C)
    #     mlin.append(1.0/np.linalg.norm(w))
    #     slin.append(len(nonzero_alpha_inds))
    #     kernel = gaussianRBFKernel
    #     w, bias, alpha, nonzero_alpha_inds = trainSVM(X_train,Y_train,kernel=kernel,C=C)
    #     mrbf.append(1.0/np.linalg.norm(w))
    #     srbf.append(len(nonzero_alpha_inds))
    # margins_lin.append(mlin)
    # sv_lin.append(slin)
    # margins_rbf.append(mrbf)
    # sv_rbf.append(srbf)



    # Plot training data/boundaries
    zz = array([predictSVM(x,X_train,Y_train,bias,alpha,kernel=kernel) for x in c_[xx.ravel(), yy.ravel()]])
    zz = zz.reshape(xx.shape)
    CS = pl.contour(xx, yy, zz, [-1, 0, 1], linestyles = 'solid', linewidths = 2)
    pl.clabel(CS, fontsize=9, inline=1)
    pl.axis('tight')

    kernel = gaussianRBFKernel
    gaussianRBF_lambda = lmbds[dataset-1]
    C = Cs[dataset-1]
    pl.subplot(2,4,dataset+4)
    pl.scatter(X_train[:, 0], X_train[:, 1], c=(1.-Y_train), s=50, cmap = pl.cm.cool)
    w, bias, alpha, nonzero_alpha_inds = trainSVM(X_train,Y_train,kernel=kernel,C=C)
    print "Trained SVM for dataset:", dataset

    # Plot training data/boundaries
    zz = array([predictSVM(x,X_train,Y_train,bias,alpha,kernel=kernel) for x in c_[xx.ravel(), yy.ravel()]])
    zz = zz.reshape(xx.shape)
    CS = pl.contour(xx, yy, zz, [-1, 0, 1], linestyles = 'solid', linewidths = 2)
    pl.clabel(CS, fontsize=9, inline=1)
    pl.axis('tight')
    print "Plotted SVM for dataset:", dataset

    # preds = predictSVM(X_train)
    # preds[preds > 0] = 1
    # preds[preds <= 0] = -1
    # errors = np.sum(np.squeeze(Y_train) != preds)
    # train_accuracy = 100*(1-errors/num_training_pts)

    # preds = predictSVM(X_val)
    # preds[preds > 0] = 1
    # preds[preds <= 0] = -1
    # errors = np.sum(np.squeeze(Y_val) != preds)
    # val_accuracy = 100*(1-errors/num_val_pts)
    # print "Dataset %i: Training Set Accuracy: %.3f, Validation Set Accuracy: %.3f" %(dataset, train_accuracy, val_accuracy)
pl.savefig("../paper/figures/2_3_testing")

pl.show()

# print margins_lin
# print '\n'
# print margins_rbf
# print '\n'
# print sv_lin
# print '\n'
# print sv_rbf

#########
# Plot margins as function of C
#########

# margins_lin = [[1.3558637135052201, 0.81655595701883565, 0.56472963174055801, 0.50234577624241938, 0.50234580597866729], [1.2447917148479888, 0.847242043878076, 0.76019679578872934, 0.7256582499273958, 0.72543466229503706], [0.93289493350784247, 0.53109161440267361, 0.2913662475935182, 0.17234490925753804, 0.11522588101990346], [3.9787718013743629, 3.367938032165255, 3.290858201447378, 3.2919100415212434, 3.297696802588697]]

# margins_rbf = [[0.12912106617913771, 0.037463120484665755, 0.02694936920036763, 0.026813216099831949, 0.026813118650490998], [0.27537441104485105, 0.067401066725651965, 0.065856907933150008, 0.060147860646297867, 0.027968708068557734], [0.26271528815265283, 0.10448608420417015, 0.069696647065049286, 0.059826442065607506, 0.050941040608137693], [3.0000313710642637, 1.3024866534418782, 0.27951634005160853, 0.21105555527448833, 0.36517345851669142]]

# pl.subplot(1,2,1)
# for i, m in enumerate(margins_lin):
#     m_normalized = [mm/m[0] for mm in m]
#     pl.semilogx(Cs,m_normalized,'--o')
# for i, m in enumerate(margins_rbf):
#     m_normalized = [mm/m[0] for mm in m]
#     pl.semilogx(Cs,m_normalized,'--o')
# pl.xlabel("C")
# pl.ylabel("Margin ($1/||w||$) (normalized)")
# # pl.savefig("../paper/figures/2_3_margins")
# # pl.show()

# sv_lin = [[75, 20, 4, 3, 3], [252, 186, 173, 173, 172], [184, 68, 33, 20, 17], [399, 393, 392, 392, 393]]

# sv_rbf = [[398, 149, 47, 49, 48], [390, 235, 124, 85, 70], [385, 135, 58, 39, 28], [400, 269, 128, 99, 86]]

# pl.subplot(1,2,2)
# for i, m in enumerate(sv_lin):
#     m_normalized = [mm/m[0] for mm in m]
#     pl.semilogx(Cs,m,'--o')
# for i, m in enumerate(sv_rbf):
#     m_normalized = [mm/m[0] for mm in m]
#     pl.semilogx(Cs,m,'--o')
# pl.xlabel("C")
# pl.ylabel("# Support Vectors")
# # pl.savefig("../paper/figures/2_3_sv")
# pl.savefig("../paper/figures/2_3_margin_sv")
# pl.show()



