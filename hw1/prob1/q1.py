import numpy as np
import loadParametersP1 as load_params

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gradient_descent(f, f_params, \
    learning_rate, convergence_limit, initial_guess, max_iters):
    # 
    # Approximate the minimum of a function minimum via gradient descent,
    # after reaching max_iters or significantly small difference
    # between iterations.
    #
    # Return x (np array of x values throughout descent)
    #         and fx (np array of f(x) values throughout descent)

    diff = np.inf
    num_iters = 0
    x = np.array([initial_guess])
    initial_fx, initial_dx = f(f_params,initial_guess)
    fx = np.array([initial_fx])
    dx = np.array([initial_dx])
    while diff > convergence_limit and num_iters < max_iters:

        next_x = x[-1] - learning_rate * dx[-1]
        
        next_fx, next_dx = f(f_params, next_x)
        x = np.vstack([x, next_x])
        fx = np.vstack([fx, next_fx])
        dx = np.vstack([dx, next_dx])
        diff = np.linalg.norm(fx[-1] - fx[-2])
        # print "diff:", diff
        num_iters += 1

    return x, fx, dx, num_iters


def negative_gaussian(params, x):
    #
    # Compute value of negative gaussian function f and its derivative d
    # Return f, d
    #
    # params should be a np array containing:
    # [mean, cov_matrix, multiplier]
    #
    # f(x) = (2x1)'*(2x2)*(2x1)
    #      = (1x2)*(2x2)*(2x1)
    #      = 1
    # df(x)/dx = 1*(2x2)*(2x1)
    #          = 2x1
    #

    mean = params[0]
    cov_matrix = params[1]
    multiplier = params[2]

    f = -(multiplier/np.sqrt(((2*np.pi)**2)*np.linalg.norm(cov_matrix))) \
        * np.exp(-0.5*np.dot(np.dot(np.transpose(x - mean), np.linalg.inv(cov_matrix)), (x - mean)))
    d = -np.dot(np.dot(f, np.linalg.inv(cov_matrix)), (x-mean))

    return f, d

def quadratic_bowl(params, x):
    #
    # Compute value of quadratic bowl function f and its derivative d
    # Return f, d
    #
    # params should be a np array containing:
    # [A, b]
    #
    # f(x) = (2x1)'*(2x2)*(2x1) - (2x1)'*(2x1)
    #      = (1x2)*(2x2)*(2x1) - (1x2)*(2x1)
    #      = 1x1 - 1x1
    #      = 1x1
    # df(x)/dx = (2x2)*(2x1) - (2x1)
    #          = (2x1)
    #

    A = params[0]
    b = params[1]

    f = 0.5 * np.dot(np.dot(np.transpose(x), A), x) \
          - np.dot(np.transpose(x), b)
    d = np.dot(A, x) - b

    return f, d

def main():
    # Load params
    gaussMean,gaussCov,quadBowlA,quadBowlb = load_params.getData()
   
    # Negative gaussian
    print "True mean:", gaussMean
    f = negative_gaussian
    multiplier = 1e4
    f_params = [gaussMean, gaussCov,multiplier]
    learning_rate = 0.01
    convergence_limit = 1e-5
    max_iters = 500

    initial_guess = np.array([8.0,8.0])

    colors = ['b','r','g','y','c','p','b']
    marker = []

    # # Plot to show effect of initial guess.
    # fig = plt.figure()
    # initial_guesses = [np.array([8.0,18.0]),np.array([12.0,16.0]),np.array([8.0,8.0])]
    # legend = []
    # for i, initial_guess in enumerate(initial_guesses):
    #     x, fx, dx, num_iters = gradient_descent(f,f_params,learning_rate,convergence_limit,initial_guess,max_iters)
    #     plt.plot(range(num_iters+1), fx, '-o', c=colors[i])
    #     legend.append('Initial Guess Error Norm: '+str(round(np.linalg.norm(initial_guess - x[-1]),2)))
    # plt.legend(legend)
    # plt.xlabel("Iteration Number")
    # plt.ylabel("f(x) [Negative Gaussian]")
    # plt.show()

    # # Plot to show effect of step size on convergence
    # fig = plt.figure()
    # convergence_limits = [1e-4,1e-3,1e-2,1e-1]
    # legend = []
    # for i, convergence_limit in enumerate(convergence_limits):
    #     x, fx, dx, num_iters = gradient_descent(f,f_params,learning_rate,convergence_limit,initial_guess,max_iters)
    #     plt.plot(range(num_iters+1), fx, '-o', c=colors[i])
    #     legend.append('Convergence Limit: '+str(convergence_limit))
    # plt.legend(legend)
    # plt.xlabel("Iteration Number")
    # plt.ylabel("f(x) [Negative Gaussian]")
    # plt.show()

    # fig = plt.figure()
    # convergence_limits = np.logspace(2,-3,20)
    # print convergence_limits
    # legend = []
    # solutions = []
    # for i, convergence_limit in enumerate(convergence_limits):
    #     x, fx, dx, num_iters = gradient_descent(f,f_params,learning_rate,convergence_limit,initial_guess,max_iters)
    #     solutions.append(fx[-1])
    # plt.semilogx(convergence_limits, solutions,'o')
    # plt.xlabel("Convergence Limit")
    # plt.ylabel("f(x) [Negative Gaussian]")
    # plt.show()

    # Quadratic Bowl
    f = quadratic_bowl
    f_params = [quadBowlA, quadBowlb]
    learning_rate = 0.01
    convergence_limit = 1e-1
    initial_guess = np.array([2.0,-2.0])
    max_iters = 20
    x, fx, dx, num_iters = gradient_descent(f,f_params,learning_rate,convergence_limit,initial_guess,max_iters)

    # print '--------'
    # print "f([10,10]):", f(f_params,np.array([10,10]))
    # print "x:", x
    # print "fx:", fx
    # print "dx:", dx
    # print "minimum: (%s,%s). Took %s steps." %(x[-1], fx[-1], np.shape(x)[0])

    # fig = plt.figure()
    # plt.scatter(range(num_iters+1), fx)
    # plt.xlabel("Iteration Number")
    # plt.ylabel("f(x)")
    # plt.show()


    fig2 = plt.figure()
    plt.plot(range(num_iters+1), np.linalg.norm(dx, axis=1),'-o')
    plt.xlabel("Iteration Number")
    plt.ylabel("Norm of Gradient [Quadratic Bowl]")
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # Z = negative_gaussian(f_params,)
    # Axes3D.plot_surface(X, Y, Z, *args, **kwargs)



if __name__ == '__main__':
    main()




