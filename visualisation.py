import matplotlib.pyplot as plt
import numpy as np

def	print_graph(km, price, m, km_mean, km_std, theta):
	plt.scatter(km, price, color='purple', marker='D') 
	plt.plot(km, np.c_[np.ones((m, 1)), (km - km_mean) / km_std].dot(theta), color='green')
	plt.xticks(rotation = 25) 
	plt.xlabel('Km') 
	plt.ylabel('Price') 
	plt.title('Graphic representation', fontsize = 20) 
	plt.show()