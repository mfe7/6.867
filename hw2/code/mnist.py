import numpy as np
import pylab as pl
from sklearn import linear_model, svm
import matplotlib.pyplot as plt

import pegasos_gaussian_test as peg
import svm_test as qp

trainSize = 200
valSize = 150
testSize = 150

def read_mnist(digit):
    name = 'data/mnist_digit_' + str(digit) + '.csv'
    data_full = np.loadtxt(name)
    train = data_full[0:trainSize, :]
    val = data_full[trainSize:(trainSize + valSize), :]
    test = data_full[(trainSize + valSize):(trainSize + valSize + testSize), :]

    return train, val, test

def normalize(data):
    return (2 * data / 255 - np.ones(np.shape(data))) 

# X is concanetated out of elements of class1 and class2
# input: e.g. class1 = np.array([1, 3, 5, 7])
def createX(class1, class2, normalize_data=True):
    classSize = class1.shape[0]

    imageSize = 784

    x_train = np.zeros((2 * classSize * trainSize, imageSize))
    x_val = np.zeros((2 * classSize * valSize, imageSize))
    x_test = np.zeros((2 * classSize * testSize, imageSize))
    y_train = np.zeros(2 * classSize * trainSize)
    y_val = np.zeros(2 * classSize * valSize)
    y_test = np.zeros(2 * classSize * testSize)

    # Add elements of first classes to X with Y = 1
    for i, digit1 in enumerate(class1):
        x_train_tmp, x_val_tmp, x_test_tmp = read_mnist(digit1)
        if normalize_data:
            x_train[i*trainSize:(i*trainSize + trainSize)] = normalize(x_train_tmp[0:trainSize])
            x_val[i*valSize:(i*valSize + valSize)] = normalize(x_val_tmp[0:valSize])
            x_test[i*testSize:(i*testSize + testSize)] = normalize(x_test_tmp[0:testSize])
        else:   
            x_train[i*trainSize:(i*trainSize + trainSize)] = x_train_tmp[0:trainSize]
            x_val[i*valSize:(i*valSize + valSize)] = x_val_tmp[0:valSize]
            x_test[i*testSize:(i*testSize + testSize)] = x_test_tmp[0:testSize]

        y_train[i*trainSize:(i*trainSize+trainSize)] = np.ones(trainSize)
        y_val[i*valSize:(i*valSize+valSize)] = np.ones(valSize)
        y_test[i*testSize:(i*testSize+testSize)] = np.ones(testSize)

    # Add elements of first classes to X with Y = -1
    for i, digit2 in enumerate(class2):
        x_train_tmp, x_val_tmp, x_test_tmp = read_mnist(digit2)
        if normalize_data:
            x_train[(classSize*trainSize) + i*trainSize : (classSize*trainSize)+(i*trainSize + trainSize)] = normalize(x_train_tmp[0:trainSize]) 
            x_val[(classSize*valSize) + i*valSize : (classSize*valSize)+(i*valSize + valSize)] = normalize(x_val_tmp[0:valSize])
            x_test[(classSize*testSize) + i*testSize : (classSize*testSize)+(i*testSize + testSize)] = normalize(x_test_tmp[0:testSize])
        else:
            x_train[(classSize*trainSize) + i*trainSize : (classSize*trainSize)+(i*trainSize + trainSize)] = x_train_tmp[0:trainSize] 
            x_val[(classSize*valSize) + i*valSize : (classSize*valSize)+(i*valSize + valSize)] = x_val_tmp[0:valSize]
            x_test[(classSize*testSize) + i*testSize : (classSize*testSize)+(i*testSize + testSize)] = x_test_tmp[0:testSize]

        y_train[(classSize*trainSize) + i*trainSize : (classSize*trainSize) + (i*trainSize+trainSize)] = -np.ones(trainSize)
        y_val[(classSize*valSize) + i*valSize : (classSize*valSize) + (i*valSize+valSize)] = -np.ones(valSize)
        y_test[(classSize*testSize) + i*testSize : (classSize*testSize) + (i*testSize+testSize)] = -np.ones(testSize)

    return x_train, x_val, x_test, y_train, y_val, y_test

def accuracy(x, y, clf):
    preds = clf.predict(x)
    incorrect_inds = np.where(preds != y)[0]
    acc = (100.0*(len(x)-len(incorrect_inds)))/len(x)
    print acc

def accuracy2(x, y, preds):
    incorrect_inds = np.where(preds != y)[0]
    acc = (100.0*(len(x)-len(incorrect_inds)))/len(x)
    print acc


def main():
    classes = [[[1],[7]],[[3],[5]],[[4],[9]],[[0,2,4,6,8],[1,3,5,7,9]]]

    lr_Cs = [1e-3, 1e-1, 1e0, 1e2]
    n_lr_Cs = [1e0, 6e1, 2e0, 1e-1]
    lr_penalties = ['l1','l1','l1','l2']
    n_lsvm_Cs = [0.2, 0.02, 0.02, 0.02]
    lsvm_Cs = [1e-6, 1e-5, 1e-5, 1e-1]

    n_svc_gammas = [0.01, 0.001, 0.001, 0.01]
    n_svc_Cs = [1, 100, 10, 1]
    svc_gammas = [1e-6, 1e-5, 0.001, 0.01]
    svc_Cs = [1e-10, 1e-10, 1e-5, 1e-5]

    for i, c in enumerate([[[4],[9]]]):
    # for i, c in enumerate(classes):
        print "Class 1:", c[0], "vs.", c[1]
        class1 = np.array(c[0])
        class2 = np.array(c[1])
        x_train, x_val, x_test, y_train, y_val, y_test = createX(class1, class2,normalize_data=False)

        # for penalty in ['l1','l2']:
        #     print "Penalty:", penalty
        #     for C in np.logspace(-10,6,num=20):
        #         clf_logReg = linear_model.LogisticRegression(penalty=penalty, dual=False,
        #                 tol=0.0001, C=C, fit_intercept=True, intercept_scaling=1,
        #                 class_weight=None, random_state=None, solver='liblinear', 
        #                 max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
        #         clf_logReg.fit(x_train, y_train)
        #         print "C:", C
        #         accuracy(x_val, y_val, clf_logReg)

        
        # # Logistic Regression
        # print "Logistic Regression:"
        # penalty = lr_penalties[i]
        # # C = lr_Cs[i]
        # C = n_lr_Cs[i]
        # clf_logReg = linear_model.LogisticRegression(penalty=penalty, dual=False,
        #         tol=0.0001, C=C, fit_intercept=True, intercept_scaling=1,
        #         class_weight=None, random_state=None, solver='liblinear', 
        #         max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
        # clf_logReg.fit(x_train, y_train)
        # print "Training acc:"
        # accuracy(x_train, y_train, clf_logReg)
        # print "Testing acc:"
        # accuracy(x_test, y_test, clf_logReg)
        # # plt.imshow(np.reshape(x_test[62],(28,28)),cmap='gray')
        # # plt.imshow(np.reshape(x_test[222],(28,28)),cmap='gray')
        # # plt.imshow(np.reshape(x_test[229],(28,28)),cmap='gray')
        # # plt.savefig('../paper/figures/4_1_bad7.png')
        # # plt.show()

        # for C in np.logspace(-10,6,num=20):
        #     clf_linear_svc = svm.LinearSVC(C=C, class_weight=None, dual=True, fit_intercept=True,
        #          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
        #          multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
        #          verbose=0)
        #     clf_linear_svc.fit(x_train, y_train) 
        #     print "C:", C
        #     accuracy(x_val, y_val, clf_linear_svc)

        # # Linear SVM
        # # C = lsvm_Cs[i]
        # C = n_lsvm_Cs[i]
        # clf_linear_svc = svm.LinearSVC(C=C, class_weight=None, dual=True, fit_intercept=True,
        #      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
        #      multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
        #      verbose=0)
        # clf_linear_svc.fit(x_train, y_train) 
        # print "Training acc:"
        # accuracy(x_train, y_train, clf_linear_svc)
        # print "Testing acc:"
        # accuracy(x_test, y_test, clf_linear_svc)

        # for gamma in np.logspace(-10,-5, num=6):
        #     print "gamma:", gamma
        #     for C in np.logspace(-10,-5,num=6):
        #         clf_rbf_svc = svm.SVC(C=C, cache_size=200, class_weight=None, coef0=0.0,
        #            decision_function_shape='ovr', degree=3, gamma=gamma, kernel='rbf',
        #            max_iter=-1, probability=False, random_state=None, shrinking=True,
        #            tol=0.001, verbose=False)
        #         clf_rbf_svc.fit(x_train, y_train)
        #         print "C:", C
        #         accuracy(x_val, y_val, clf_rbf_svc) 

        
        # # Linear SVM
        # # C = n_svc_Cs[i]
        # # gamma = n_svc_gammas[i]
        # C = svc_Cs[i]
        # gamma = svc_gammas[i]
        # clf_rbf_svc = svm.SVC(C=C, cache_size=200, class_weight=None, coef0=0.0,
        #    decision_function_shape='ovr', degree=3, gamma=gamma, kernel='rbf',
        #    max_iter=-1, probability=False, random_state=None, shrinking=True,
        #    tol=0.001, verbose=False)
        # clf_rbf_svc.fit(x_train, y_train)
        # accuracy(x_train, y_train, clf_rbf_svc)
        # accuracy(x_test, y_test, clf_rbf_svc)


        # # Pegasos
        # lmbda = .02;
        # epochs = 100;
        # for gamma in np.logspace(-10,-5, num=6):
        #     print "gamma:", gamma
        #     K = peg.gaussian_rbf_matrix(x_train, gamma)
        #     alpha = peg.train_gaussianSVM(x_train,y_train, K, lmbda, epochs)
        #     preds_val = peg.predict_gaussianSVM(x_val, alpha, x_train, gamma)
        #     accuracy2(x_val, y_val, preds_val)

        # QP
        epochs = 100;
        gaussianRBF_lambda = 0.2
        for C in np.logspace(-10,-5, num=6):
            kernel = qp.gaussianRBFKernel
            w, bias, alpha, nonzero_alpha_inds = qp.trainSVM(x_train,y_train,kernel=kernel,C=C)
            preds_val = peg.predictSVM(x_val,x_train,y_train,bias,alpha,kernel=kernel)
            accuracy2(x_val, y_val, preds_val) 


        # # C = n_svc_Cs[i]
        # # gamma = n_svc_gammas[i]
        # C = svc_Cs[i]
        # gamma = svc_gammas[i]
        # clf_rbf_svc = svm.SVC(C=C, cache_size=200, class_weight=None, coef0=0.0,
        #    decision_function_shape='ovr', degree=3, gamma=gamma, kernel='rbf',
        #    max_iter=-1, probability=False, random_state=None, shrinking=True,
        #    tol=0.001, verbose=False)
        # clf_rbf_svc.fit(x_train, y_train)
        # accuracy(x_train, y_train, clf_rbf_svc)
        # accuracy(x_test, y_test, clf_rbf_svc)
         



if __name__ == '__main__':
    main()
