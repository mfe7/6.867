import numpy as np
import pylab as pl
import sklearn.linear_model
# from sklearn import linear_model

def sgd_train(X,Y, penalty):
    sgd = sklearn.linear_model.SGDClassifier(penalty=penalty)
    sgd.fit(X, np.ravel(Y))
    return sgd.coef_, sgd.intercept_

def lr_train(X,Y,penalty, lamb):
    LR = sklearn.linear_model.LogisticRegression(penalty=penalty, dual=False, tol=0.0001, C=1.0/lamb, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    LR.fit(X,np.ravel(Y))
    return LR.coef_, LR.intercept_
