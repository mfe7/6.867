import numpy as np
import loadParametersP1 as load_params

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

    return x, fx


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
    initial_guess = np.array([8.0,18.0])
    max_iters = 10000
    x, fx = gradient_descent(f,f_params,learning_rate,convergence_limit,initial_guess,max_iters)

    # # Quadratic Bowl
    # f = quadratic_bowl
    # f_params = [quadBowlA, quadBowlb]
    # learning_rate = 0.01
    # convergence_limit = 1e-1
    # initial_guess = np.array([2.0,-2.0])
    # max_iters = 20
    # x, fx = gradient_descent(f,f_params,learning_rate,convergence_limit,initial_guess,max_iters)

    print '--------'
    print "f([10,10]):", f(f_params,np.array([10,10]))
    print "x:", x
    print "fx:", fx
    print "minimum: (%s,%s). Took %s steps." %(x[-1], fx[-1], np.shape(x)[0])

if __name__ == '__main__':
    main()




