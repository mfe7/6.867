import numpy as np
import random
import loadFittingDataP1 as load_data
import q1, q2

def sgd(f, f_params, \
    learning_rate, convergence_limit, initial_guess, max_iters):
    # 
    # Approximate the minimum of a function minimum via stochastic gradient descent,
    # after reaching max_iters or significantly small difference
    # between iterations.
    #
    # Return x (np array of x values throughout descent)
    #         and fx (np array of f(x) values throughout descent)

    # SGD params (learning rate schedule)
    t0 = 1e10
    k = 0.6

    diff = np.inf
    num_iters = 0
    x = np.array([initial_guess])
    initial_fx, initial_dx = f(f_params,initial_guess)
    fx = np.array([initial_fx])
    dx = np.array([initial_dx])
    while diff > convergence_limit and num_iters < max_iters:
        learning_rate = (t0 + num_iters)**(-k)
        next_x = x[-1] - learning_rate * dx[-1]
        next_fx, next_dx = f(f_params, next_x)
        x = np.vstack([x, next_x])
        fx = np.vstack([fx, next_fx])
        dx = np.vstack([dx, next_dx])
        diff = np.linalg.norm(fx[-1] - fx[-2])
        # print "diff:", diff
        num_iters += 1

    return x, fx, num_iters

def lsq_gradient(Theta, xi, yi):
    #
    # Given a single (x,y) pair from the dataset,
    # calculate gradient of lsq error.
    #

    grad = 2*np.dot(np.dot(np.transpose(xi), Theta)-yi, np.transpose(xi))

    return grad

def lsq_error(f_params, Theta):
    X = f_params[0]
    y = f_params[1]
    # operates on entire dataset
    J = np.linalg.norm(np.dot(X, Theta) - y)**2
    return J


def least_square_error(f_params, Theta):
    # calculate lsq error directly
    # and gradient from single, randomly
    # selected (x,y pair).
    X = f_params[0]
    y = f_params[1]
    J = lsq_error(f_params, Theta)
    
    i = random.randint(0,np.shape(X)[0]-1)
    random_x = X[i]
    random_y = y[i]
    g = lsq_gradient(Theta, random_x, random_y)

    return J, g

def lsq(f_params, x):
    # Calculate lsq error directly
    # and approximate gradient via entire dataset
    J, _ = least_square_error(f_params, x)
    dx = 0.0001
    grad = q2.gradient_approximation(least_square_error, f_params, x, dx)
    return J, grad

def main():

    # Set up Least Squares w/ data
    X, y = load_data.getData()
    f = lsq
    f_params = [X, y]

    # # Analytic solution
    # Theta_pinv = np.dot(np.linalg.pinv(X),y)
    # J_Theta_pinv, _ = lsq(f_params, Theta_pinv)
    # print "-------------"
    # print "Analytic:"
    # print "Theta: %s" %(Theta_pinv)
    # print "J(Theta): %s" %(J_Theta_pinv)
    # print "-------------"

    # # Batch GD params
    # learning_rate = 0.00001
    # convergence_limit = 1e-5
    # initial_guess = np.zeros([np.shape(X)[1]])
    # max_iters = 10000

    # Theta_BGD, J_Theta_BGD, num_iters_BGD = q1.gradient_descent(f,f_params,learning_rate,convergence_limit,initial_guess,max_iters)
    # print "-------------"
    # print "Batch GD:"
    # print "Theta: %s" %(Theta_BGD[-1])
    # print "J(Theta): %s" %(J_Theta_BGD[-1])
    # print "Took %s steps." %num_iters_BGD
    # print "-------------"

    # SGD params
    f = least_square_error
    learning_rate = 0.00001
    convergence_limit = 1e-5
    initial_guess = np.zeros([np.shape(X)[1]])
    max_iters = 10000

    Theta_SGD, J_Theta_SGD, num_iters_SGD = sgd(f,f_params,learning_rate,convergence_limit,initial_guess,max_iters)
    print "-------------"
    print "Stochastic GD:"
    print "Theta: %s" %(Theta_SGD[-1])
    print "J(Theta): %s" %(J_Theta_SGD[-1])
    print "Took %s steps." %num_iters_SGD
    print "-------------"
    
    return

if __name__ == '__main__':
    main()