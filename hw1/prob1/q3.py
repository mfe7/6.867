import numpy as np
import loadFittingDataP1 as load_data
import q1, q2

def batch_gd():
	return

def least_square_error(f_params, Theta):
	X = f_params[0]
	y = f_params[1]
	J = np.linalg.norm(np.dot(X, Theta) - y)**2
	return J

def lsq(f_params, x, dx):
	q2.gradient_approximation(least_square_error, f_params, x, dx)
	return J, grad

def main():
	X, y = load_data.getData()

	return

if __name__ == '__main__':
	main()