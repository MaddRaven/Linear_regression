def	predict(mileage, theta0, theta1):
	return theta0 + (theta1 * mileage)

theta0 = 0
theta1 = 0
mileage = input("Enter mileage: ")
try:
	mileage = float(mileage)
	predict = predict(mileage, theta0, theta1)
	print(f"Price estimated: {predict:.2f}")
except ValueError:
	print("Incorrect input")
