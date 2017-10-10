import numpy as np
import loadFittingDataP2 as load_data
import matplotlib.pyplot as plt
import 

def sum_of_squares_error(f_params,w):
    X = f_params[0]
    y = f_params[1]
    M = f_params[2]

    phi_X = polynomial_basis(X, M)
    error = np.sum((y - np.dot(np.transpose(w),phi_X))**2)
    # grad = 
    return e

def polynomial_basis(X, M):
    phi_X = np.empty([np.shape(X)[0], M+1])
    for i in range(M+1):
        phi_X[:,i] = X**i
    return phi_X

def cosine_basis(X, M):
    phi_X = np.empty([np.shape(X)[0], M+1])
    phi_X[:,0] = np.ones([np.shape(X)[0]])
    for i in range(1,M+1):
        phi_X[:,i] = np.cos(np.pi*X*i)
    return phi_X

def max_likelihood(X, y, M, f_basis, regularizer=0):
    phi_X = f_basis(X, M)
    w = np.dot(np.dot(np.linalg.inv(regularizer*np.eye(M+1,M+1)+np.dot(np.transpose(phi_X),phi_X)),np.transpose(phi_X)),y)
    return w


def main():
    X, y = load_data.getData()

    # Batch GD
    f = sum_of_squares_error
    M = 2
    f_params = [X, y, M]
    learning_rate = 0.01
    convergence_limit = 1e-1
    initial_guess = np.array([2.0,-2.0])
    max_iters = 20
    x, fx, dx, num_iters = gradient_descent(f,f_params,learning_rate,convergence_limit,initial_guess,max_iters)


    # # Plot different model complexities
    # plot_X = np.linspace(0,1,100)
    # true_y = np.cos(np.pi*plot_X)+1.5*np.cos(2*np.pi*plot_X)

    # M = 2
    # cosine_plot_X = cosine_basis(plot_X, M)

    # M = 4
    # polynomial_plot_X = polynomial_basis(plot_X, M)


    # M = 4
    # f_basis = polynomial_basis
    # w_polynomial = max_likelihood(X, y, M, f_basis)

    # M = 2
    # f_basis = cosine_basis
    # w_cosine = max_likelihood(X, y, M, f_basis, regularizer=0)
    # print w_cosine


    # y_polynomial = np.dot(w_polynomial, np.transpose(polynomial_plot_X))
    # y_cosine = np.dot(w_cosine, np.transpose(cosine_plot_X))

    # fig = plt.figure()
    # # for i, M in enumerate([0,1,2,10]):
    # for i, M in enumerate([1,2,4,8]):
    #     cosine_plot_X = cosine_basis(plot_X, M)
    #     polynomial_plot_X = polynomial_basis(plot_X, M)

    #     f_basis = polynomial_basis
    #     w_polynomial = max_likelihood(X, y, M, f_basis)
    #     y_polynomial = np.dot(w_polynomial, np.transpose(polynomial_plot_X))
        
    #     f_basis = cosine_basis
    #     w_cosine = max_likelihood(X, y, M, f_basis, regularizer=0)
    #     y_cosine = np.dot(w_cosine, np.transpose(cosine_plot_X))
    #     # print [round(i,2) for i in w_cosine]

    # #     p = plt.subplot(1,4,i+1)
    # #     plt.plot(X,y,'co', markersize=10)
    # #     plt.plot(plot_X, true_y, 'g-')
    # #     plt.plot(plot_X, y_polynomial, 'r-')
    # #     plt.plot(plot_X, y_cosine, 'b-')
    # #     plt.xlabel('x')
    # #     plt.ylabel('y')
    # #     plt.title('Linear Regression (M=%i)'%M)
    # # plt.show()

    return

if __name__ == '__main__':
    main()