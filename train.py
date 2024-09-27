import csv
import matplotlib.pyplot as plt 

km = [] 
price = [] 

try:
	with open('data.csv', 'r') as file:
		reader = csv.reader(file, delimiter=',')
		next(reader)
		for row in reader:
			km.append(int(row[0]))
			price.append(int(row[1])) 

		plt.scatter(km, price, color='purple', marker='D') 
		plt.xticks(rotation = 25) 
		plt.xlabel('Km') 
		plt.ylabel('Price') 
		plt.title('Graphic representation', fontsize = 20) 
		plt.show()

except FileNotFoundError:
	print("File not found")
