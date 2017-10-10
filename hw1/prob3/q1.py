import numpy as np
import matplotlib.pyplot as plt


def ridge_regression(X, y, M, f_basis, regularizer):
    phi_X = f_basis(X, M)
    w = np.dot(np.dot(np.linalg.inv(regularizer*np.eye(M+1,M+1)+np.dot(np.transpose(phi_X),phi_X)),np.transpose(phi_X)),y)
    return w

def polynomial_basis(X, M):
    phi_X = np.empty([np.shape(X)[0], M+1])
    for i in range(M+1):
        phi_X[:,i] = X**i
    return phi_X

def compare_M_lambda():
    fig = plt.figure()
    for i, M in enumerate([0,1,2,10]):
        plt.subplot(1,4,i+1)
        # plt.plot(X,y,'co', markersize=10)
        polynomial_plot_X = polynomial_basis(plot_X, M)
        f_basis = polynomial_basis
        for l in [0,0.5,1,10]:
            w_polynomial = max_likelihood(X, y, M, f_basis)
            y_polynomial = np.dot(w_polynomial, np.transpose(polynomial_plot_X))
            plt.plot(plot_X, y_polynomial, 'r-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Regression (M=%i)'%M)
    plt.show()


def main():
    compare_M_lambda()
    return

if __name__ == '__main__':
    main()