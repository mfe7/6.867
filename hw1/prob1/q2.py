import numpy as np
import loadParametersP1 as load_params
import q1
import matplotlib.pyplot as plt


def gradient_approximation(f, f_params, x, dx):
    #
    # Use Finite Difference method to numerically 
    # evaluate the gradient at a point
    #
    # g ~= (f(x+dx)-f(x))/dx

    g = np.empty([len(x)])
    for i, xi in enumerate(x):
        xdx = np.copy(x)
        xdx[i] = xdx[i] + dx
        fxdx = f(f_params, xdx)
        fx = f(f_params, x)
        if type(fxdx) == tuple:
            fxdx = fxdx[0]
            fx = fx[0]
        g[i] = (fxdx - fx) / dx

    return g


def main():
    # Load params
    gaussMean,gaussCov,quadBowlA,quadBowlb = load_params.getData()
   
    # Negative gaussian
    f = q1.negative_gaussian
    multiplier = 1e4
    f_params = [gaussMean, gaussCov,multiplier]

    # Gradient params
    x = np.array([3.0,4.0])
    dx = 0.01

    # Calculate real and approx gradients
    real_gradient = f(f_params, x)[1]
    approx_gradient = gradient_approximation(f, f_params, x, dx)
    gradient_error = approx_gradient - real_gradient
    error_norm = np.linalg.norm(gradient_error)
    print x
    print "True: %s, Approx: %s, Error: %s, Error Norm: %s" %(real_gradient, approx_gradient, gradient_error, error_norm)
    print "------"

    x = np.array([3.0,4.0])
    error_norms = []
    dxs = np.logspace(0,-5)
    for dx in dxs:
        real_gradient = f(f_params, x)[1]
        approx_gradient = gradient_approximation(f, f_params, x, dx)
        gradient_error = approx_gradient - real_gradient
        error_norm = np.linalg.norm(gradient_error)
        error_norms.append(error_norm)
    fig = plt.figure()
    plt.semilogx(dxs, error_norms, '-o')
    plt.xlabel('$\delta x$: Difference Step')
    plt.ylabel('Norm of Error between Real and Approximate Gradient')
    plt.show()


    # Gradient params
    x = np.array([12.0,4.0])
    dx = 0.01

    # Calculate real and approx gradients
    real_gradient = f(f_params, x)[1]
    approx_gradient = gradient_approximation(f, f_params, x, dx)
    gradient_error = approx_gradient - real_gradient
    error_norm = np.linalg.norm(gradient_error)
    print x
    print "True: %s, Approx: %s, Error: %s, Error Norm: %s" %(real_gradient, approx_gradient, gradient_error, error_norm)
    print "------"


    # Quadratic Bowl
    f = q1.quadratic_bowl
    f_params = [quadBowlA, quadBowlb]

    # Gradient params
    x = np.array([6.0,2.0])
    dx = 0.01

    # Calculate real and approx gradients
    real_gradient = f(f_params, x)[1]
    approx_gradient = gradient_approximation(f, f_params, x, dx)
    gradient_error = approx_gradient - real_gradient
    error_norm = np.linalg.norm(gradient_error)
    print x
    print "True: %s, Approx: %s, Error: %s, Error Norm: %s" %(real_gradient, approx_gradient, gradient_error, error_norm)
    print "------"

    # Gradient params
    x = np.array([1000.0,90.0])
    dx = 0.01

    # Calculate real and approx gradients
    real_gradient = f(f_params, x)[1]
    approx_gradient = gradient_approximation(f, f_params, x, dx)
    gradient_error = approx_gradient - real_gradient
    error_norm = np.linalg.norm(gradient_error)
    print x
    print "True: %s, Approx: %s, Error: %s, Error Norm: %s" %(real_gradient, approx_gradient, gradient_error, error_norm)
    print "------"


if __name__ == '__main__':
    main()