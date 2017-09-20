import numpy as np
from sklearn import linear_model
import lassoData
import matplotlib.pyplot as plt

def phi(X):
    feature_vector_length = 13
    phi_X = np.empty([np.shape(X)[0], feature_vector_length])
    phi_X[:,0] = X
    # phi_X[:,0] = X[:,0]
    for i in range(1,feature_vector_length):
        # phi_X[:,i] = np.sin(0.4*np.pi*X[:,0]*i)
        phi_X[:,i] = np.sin(0.4*np.pi*X*i)
    return phi_X

def compare_lasso_regularizer(phi_train_X, train_y, phi_plot_X):
    alphas = [0.01,0.05,0.1,0.5,1.0,10.0,100.0]
    legend = []
    for alpha in alphas:
        # fit lasso model
        clf = linear_model.Lasso(alpha=alpha)
        clf.fit(phi_train_X, train_y)
        print clf.coef_
        # print clf.intercept_
        plt.plot(range(len(clf.coef_)), clf.coef_,'-x')
        legend.append(r'$\lambda$='+str(alpha))
        # y_lasso = np.dot(clf.coef_, np.transpose(phi_plot_X))
    plt.legend(legend)
    plt.xlabel('Element of weight vector')
    plt.ylabel('Value of weight vector element (LASSO)')
    plt.show()






def main():
    # Load the data
    train_X, train_y = lassoData.lassoTrainData()
    test_X, test_y = lassoData.lassoTestData()
    validate_X, validate_y = lassoData.lassoValData()

    # Convert raw X data to feature vectors
    phi_train_X = phi(train_X[:,0])
    plot_X = np.linspace(-1,1,100)
    phi_plot_X = phi(plot_X)

    # fit lasso model
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(phi_train_X, train_y)
    y_lasso = np.dot(clf.coef_, np.transpose(phi_plot_X))

     # fit lasso model
    clf = linear_model.Lasso(alpha=1.0)
    clf.fit(phi_train_X, train_y)
    y_lasso2 = np.dot(clf.coef_, np.transpose(phi_plot_X))


    # Make a plot of lasso output wrt changes in regularization
    compare_lasso_regularizer(phi_train_X, train_y, phi_plot_X)

    # Plot all the data, and the models on one set of axes
    plt.plot(train_X, train_y, 'o')
    plt.plot(test_X, test_y, 'o')
    plt.plot(validate_X, validate_y, 'o')
    plt.plot(plot_X, y_lasso, '-')
    plt.plot(plot_X, y_lasso2, '-')

    plt.legend(['Training Data', 'Test Data', 'Validation Data', 'Lasso '+r'($\lambda=0.1$)', 'Lasso '+r'($\lambda=1.0$)'])
    plt.show()



if __name__ == '__main__':
    main()
