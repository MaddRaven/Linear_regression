import os.path
import numpy as np

def	predict(mileage, theta0, theta1, norm_mean, norm_std):
	norm_mil = (mileage - norm_mean) / norm_std
	return theta0 + theta1 * norm_mil


def main():
	while True:
		try:
			mileage = input("Enter mileage: ")

			try:
				mileage = float(mileage)
			except ValueError:
				print("Incorrect input")
				exit(1)

			file = 'trained_params.txt'

			if os.path.isfile(file) == False:
				theta0 = 0
				theta1 = 0
				norm_mean = 0
				norm_std = 1
			else:
				try:
					theta0, theta1, norm_mean, norm_std = np.loadtxt('trained_params.txt')
				except Exception as e:
					print("Error extracting datas")
					exit(1)

			prediction = predict(mileage, theta0, theta1, norm_mean, norm_std)
			if prediction < 0:
				print("Mileage too high")
			elif mileage < 0:
				print("Mileage should be positive")
			else:
				print(f"Price estimated: {prediction:.2f}")

		except KeyboardInterrupt:
			print("\nProgram interrupted")
			break


if __name__ == "__main__":
    main()