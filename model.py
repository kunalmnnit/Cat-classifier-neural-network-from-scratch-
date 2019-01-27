import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from helper_functions import *

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#layers_dims = [12288, 20, 7, 5, 1]

m_x = train_set_x_orig.shape[1]

# flatten the images
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0] , -1).T 
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0] , -1).T 

#preprocessing
train_set_x = train_set_x_flatten / 255 
test_set_x  = test_set_x_flatten / 255

def model(X , Y  , learning_rate , num_iterations , layer_dims):

	np.random.seed(1)
	costs = []

	parameters = initialize_parameters_deep(layer_dims)
	

	for i in range(num_iterations):
		AL , caches = forward_prop(X , parameters)

		cost = compute_cost(AL , Y)

		grads = backward_prop(AL , Y , caches)

		parameters = update_parameters(parameters , grads , learning_rate)

		if i % 100 == 0:
			costs.append(cost)
			print("cost after {} iteration {}:".format(i,cost))

	#plot the cost
	cost = np.squeeze(costs)
	plt.plot(cost)
	plt.ylabel('cost')
	plt.xlabel('iterations(per 100)')
	plt.title("learning rate"+str(learning_rate))
	plt.show()

	return parameters

parameters = model(train_set_x , train_set_y , learning_rate = 0.0075  , num_iterations= 2501 , layer_dims =[12288, 20, 7 , 5 ,1])

pred_train = accuracy(train_set_x , train_set_y , parameters)
pred_test = accuracy(test_set_x , test_set_y , parameters)


myimage = "muffu.jpg"
fname = "images/"+myimage
image = np.array(ndimage.imread(fname, flatten = False))
myimage =scipy.misc.imresize(image , size=(m_x , m_x)).reshape((1 , m_x*m_x*3)).T
my_predicted_image = predict(myimage , parameters)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

