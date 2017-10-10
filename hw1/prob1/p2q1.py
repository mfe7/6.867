import numpy as np
import loadFittingDataP2 as load_data
import matplotlib.pyplot as plt
import q1,q2,q3

def lsq_error(f_params,w):
    X = f_params[0]
    y = f_params[1]
    M = f_params[2]

    phi_X = polynomial_basis(X, M)
    error = np.sum((y - np.dot(phi_X,np.transpose(w)))**2)
    # e = (y - np.dot(phi_X,np.transpose(w)))
    # error = np.inner(e,e)
    return error

def sum_of_squares_error(f_params,w,ii=None):

    X = f_params[0]
    y = f_params[1]
    M = f_params[2]
    phi_X = polynomial_basis(X, M)

    error = lsq_error(f_params,w)
    dx = 0.001


    # if ii is None:
    #     grad1 = -2*np.dot(phi_X.T, y) + 2*np.dot(np.dot(phi_X.T, phi_X), w)
        
    # else:
    #     random_x = X[ii]
    #     random_y = y[ii]
    #     phi_random_x = polynomial_basis(random_x,M)
    #     print phi_random_x, random_y
    #     grad1 = -2*np.dot(phi_random_x, random_y)
    #     print np.shape(grad1)
    #     grad2 = 2*np.dot(np.dot(phi_random_x.T, phi_random_x),w.T)
    #     print np.shape(grad1)
    #     grad = grad1+grad2


    grad = -2*np.dot(phi_X.T, y) + 2*np.dot(np.dot(phi_X.T, phi_X), w)
    grad2 = q2.gradient_approximation(lsq_error, f_params, w, dx)
    norm = np.linalg.norm(grad-grad2)

    return error, grad

def polynomial_basis(X, M):
    if np.ndim(X) == 0:
        N = 1
    else:
        N = np.shape(X)[0]
    phi_X = np.empty([N, M+1])
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


    # # Batch GD
    # f = sum_of_squares_error
    # M = 2
    # # phi_X = polynomial_basis(X,M)
    # f_params = [X, y, M]
    # learning_rate = 0.01
    # convergence_limit = 1e-3
    # initial_guess = np.array([5, -5, 5])
    # # initial_guess = np.zeros([M+1])
    # max_iters = 100
    # x, fx, dx, num_iters = q1.gradient_descent(f,f_params,learning_rate,convergence_limit,initial_guess,max_iters)
    # w_bgd = x[-1]
    # print x

    # # SGD
    # f = sum_of_squares_error
    # M = 2
    # f_params = [X, y, M]
    # learning_rate = 0.01
    # convergence_limit = 1e-8
    # initial_guess = np.zeros([M+1])
    # # guesses = [2,-10,10]
    # # for ii in range(min(M,3)):
    # #     initial_guess[ii] = guesses[ii]
    # max_iters = 10000
    # t0 = 1e8
    # k = 0.6
    # w_sgds, fx, num_iters = q3.sgd(f,f_params,convergence_limit,initial_guess,max_iters,t0,k)
    # print fx, num_iters
    # w_sgd = w_sgds[-1]
    # print w_sgd
    # # y_sgd = np.dot(w_sgd, np.transpose(polynomial_plot_X))

    # errors = []
    # for xx in dx:
    #     _,_,n = sum_of_squares_error(f_params,xx,0.01)
    #     errors.append(n)
    # print errors
    # print "max error:", max(errors)
    # print "min error:", min(errors)
    # print "mean error:", np.mean(errors)
    # print "num errors:", len(errors)
    # # sse_bgd = lsq_error(f_params,x[-1])
    # # print "BGD SSE:", sse_bgd


    # Plot different model complexities
    plot_X = np.linspace(0,1,100)
    true_y = np.cos(np.pi*plot_X)+np.cos(2*np.pi*plot_X)

    # # M = 2
    # # cosine_plot_X = cosine_basis(plot_X, M)

    # # M = 4
    # polynomial_plot_X = polynomial_basis(plot_X, M)
    # y_bgd = np.dot(w_bgd, np.transpose(polynomial_plot_X))
    # fig = plt.figure()
    # print polynomial_plot_X
    # plt.plot(plot_X,y_bgd,'r-')
    # plt.plot(plot_X,true_y,'g-')
    # plt.show()

    # M = 2
    # f_basis = polynomial_basis
    # w_polynomial = max_likelihood(X, y, M, f_basis)
    # sse_polynomial = lsq_error(f_params,w_polynomial)
    # # print "Polynomial SSE:", sse_polynomial
    # # print "w_poly:", w_polynomial

    # # M = 2
    # # f_basis = cosine_basis
    # # w_cosine = max_likelihood(X, y, M, f_basis, regularizer=0)
    # # print w_cosine


    # # y_polynomial = np.dot(w_polynomial, np.transpose(polynomial_plot_X))
    # # y_cosine = np.dot(w_cosine, np.transpose(cosine_plot_X))

    fig = plt.figure()
    # for i, M in enumerate([0,1,2,10]):
    for i, M in enumerate([1,2,4,8]):
        cosine_plot_X = cosine_basis(plot_X, M)
        polynomial_plot_X = polynomial_basis(plot_X, M)

        f_basis = polynomial_basis
        w_polynomial = max_likelihood(X, y, M, f_basis)
        y_polynomial = np.dot(w_polynomial, np.transpose(polynomial_plot_X))


        f_basis = cosine_basis
        w_cosine = max_likelihood(X, y, M, f_basis, regularizer=0)
        y_cosine = np.dot(w_cosine, np.transpose(cosine_plot_X))

        # Batch GD
        f = sum_of_squares_error
        f_params = [X, y, M]
        learning_rate = 0.01
        convergence_limit = 1e-3
        initial_guess = np.zeros([M+1])
        guesses = [2,-10,10]
        for ii in range(min(M,3)):
            initial_guess[ii] = guesses[ii]
        max_iters = 100
        w_bgds, fx, dx, num_iters = q1.gradient_descent(f,f_params,learning_rate,convergence_limit,initial_guess,max_iters)
        w_bgd = w_bgds[-1]
        y_bgd = np.dot(w_bgd, np.transpose(polynomial_plot_X))

        # SGD
        f = sum_of_squares_error
        f_params = [X, y, M]
        convergence_limit = 1e-8
        initial_guess = np.zeros([M+1])
        guesses = [2,-10,10]
        for ii in range(min(M,3)):
            initial_guess[ii] = guesses[ii]
        max_iters = 10000
        t0 = 1e8
        k = 0.6
        w_sgds, fx, num_iters = q3.sgd(f,f_params,convergence_limit,initial_guess,max_iters,t0,k)
        w_sgd = w_sgds[-1]
        y_sgd = np.dot(w_sgd, np.transpose(polynomial_plot_X))

        p = plt.subplot(1,4,i+1)
        plt.plot(X,y,'co', markersize=10,markerfacecolor='none')
        plt.plot(plot_X, true_y, 'g-')
        plt.plot(plot_X, y_polynomial, '-',color='orange')
        plt.plot(plot_X, y_cosine, 'b-')
        plt.plot(plot_X, y_bgd, 'm-')
        plt.plot(plot_X, y_sgd, '-', color='black')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Regression (M=%i)'%M)
    plt.legend(['Data','True','Polynomial','Cosine','GD','SGD'])
    plt.show()

    return

if __name__ == '__main__':
    main()