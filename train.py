import csv
import matplotlib.pyplot as plt
import numpy as np


def	print_graph(km, price):
	plt.scatter(km, price, color='orange', marker='D') 
	plt.xticks(rotation = 25) 
	plt.xlabel('Km') 
	plt.ylabel('Price') 
	plt.title('Graphic representation', fontsize = 20) 
	plt.show()


def print_norm_graph(norm_km, price):
    plt.scatter(norm_km, price, color='purple', marker='D') 
    plt.xlabel('Normalized Km') 
    plt.ylabel('Price') 
    plt.title('Normalized Graphic Representation', fontsize = 20) 
    plt.show()


def normalize(data):
    min_data = min(x for x in data if x != 0)
    max_data = max(x for x in data if x != 0)
    
    valid_data = [x for x in data if x != 0 and (min_data <= x <= max_data)]
    
    if not valid_data:
        return data
    
    normalized_data = [(x - min_data) / (max_data - min_data) for x in valid_data]
    
    norm_data = []
    for i in range(len(data)):
        if data[i] != 0 and (min_data <= data[i] <= max_data):
            norm_data.append(normalized_data[len(norm_data)])
    
    return norm_data


def model(theta0, theta1, norm_km, norm_price):
	return np.sum((theta0 + theta1 * norm_km) - norm_price)


def train_model(norm_km, norm_price):
	m = len(norm_km)
	learning_rate = 0.01

	theta0 = 0
	theta1 = 0
	
	norm_km = np.array(norm_km)
	norm_price = np.array(norm_price)

	for _ in range(1000):
		gradient_theta0 = learning_rate * model(theta0, theta1, norm_km, norm_price) * (1/m)
		gradient_theta1 = learning_rate * model(theta0, theta1, norm_km, norm_price) * np.sum(theta0 + theta1 * norm_km) * (1/m)
           
		theta0 -= gradient_theta0
		theta1 -= gradient_theta1

	return theta0, theta1


def	main():
	km = [] 
	price = [] 

	try:
		with open('data.csv', 'r') as file:
			reader = csv.reader(file, delimiter=',')
			next(reader)
			for row in reader:
				km.append(int(row[0]))
				price.append(int(row[1])) 
			print_graph(km, price)
			norm_km = normalize(km)
			norm_price = normalize(price)
			print_norm_graph(norm_km, norm_price)

	except FileNotFoundError:
		print("File not found")

	theta0, theta1 = train_model(norm_km, norm_price)
	with open('trained_params.txt', 'w') as f:
	    f.write(f"{theta0}\n{theta1}")

if __name__ == "__main__":
    main()