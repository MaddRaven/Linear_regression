import numpy as np
from print import print_graph

def train_model(x, y, x_mean, x_std):
	learning_rate = 0.01
	iterations = 1000
	theta0 = 0
	theta1 = 0
	m = len(x)
	
	for _ in range(iterations):
		estimation = theta0 + theta1 * ((x - x_mean) / x_std)

		grad_theta0 = (np.sum(estimation - y) * learning_rate) / m
		grad_theta1 = (np.sum((estimation - y) * ((x - x_mean) / x_std)) * learning_rate) / m

		theta0 -= grad_theta0
		theta1 -= grad_theta1

	return theta0, theta1


def	main():
	try:
		data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)
	except Exception as e:
		print("Error extracting datas")
		exit(1)

	x = data[:, 0].reshape(-1, 1)
	y = data[:, 1].reshape(-1, 1)
	x_mean = np.mean(x, axis=0)
	x_std = np.std(x, axis=0)

	theta0, theta1 = train_model(x, y, x_mean, x_std)

	np.savetxt('trained_params.txt', [theta0, theta1, x_mean, x_std], fmt='%f')

	m = len(x)
	theta = [theta0, theta1]
	print_graph(x, y, m, x_mean, x_std, theta)

if __name__ == "__main__":
    main()