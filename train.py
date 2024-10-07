import numpy as np
from print import print_graph
from predict import predict

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


def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    
    if abs(ss_tot) < 1e-10:
        return 0
    
    return 1 - (ss_res / ss_tot)


def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


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

	np.savetxt('trained_params.txt', [[theta0], [theta1], x_mean, x_std])

	y = data[:, 1]
	y_pred = []
	for _ in range(len(x)):
		y_pred.append(predict(x[_], theta0, theta1, x_mean, x_std)[0])

	mae = calculate_mae(np.sum(y), np.sum(y_pred))
	rmse = calculate_rmse(np.sum(y), np.sum(y_pred))
	r2 = r2_score(np.sum(y), np.sum(y_pred))
	mse = calculate_mse(np.sum(y), np.sum(y_pred))

	print(f"MAE: {mae}")
	print(f"RMSE: {rmse}")
	print(f"R-Squared: {r2}")
	print(f"MSE: {mse}")

	m = len(x)
	theta = [theta0, theta1]
	print_graph(x, y, m, x_mean, x_std, theta)


if __name__ == "__main__":
    main()