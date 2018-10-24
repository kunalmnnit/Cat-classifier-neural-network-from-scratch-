import numpy as np
import h5py
import matplotlib.pyplot as plt

def initialize_parameters(n_x , n_h , n_y):
	W1 = np.random.randn(n_h , n_x) * 0.01
	b1 = np.zeros([n_h , 1])
	W2 = np.random.randn(n_y , n_h) * 0.01
	b2 = np.zeros([n_y , 1])

	parameters = {"W1":W1 , "b1":b1 , "W2":W2 , "b2":b2}
	return parameters

def initialize_parameters_deep(layer_dims):
	np.random.seed(1)
	parameters = {}
	L = len(layer_dims)            # number of layers in the network

	for l in range(1, L):
		parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
		#parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
		parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
	return parameters

def sigmoid(Z):

	cache = Z
	A= 1/(1+np.exp(-Z))

	return A , cache

def relu(Z):

	cache = Z
	A = np.maximum(0 , Z)

	return A , cache

def backward_relu(dA , cache):

	Z = cache
	dZ = np.array(dA ,copy = True)
	dZ[Z<=0] = 0

	return dZ

def backward_sigmoid(dA , cache):

	Z = cache
	s = 1/(1+np.exp(-Z))
	dZ = dA * s * (1-s)

	return dZ

def linear_forward(A , W , b ):
	Z = np.dot(W , A) + b
	cache = (A, W, b)
	return Z, cache

def linear_forward_deep(A_prev, W , b , activation):

	if activation == "sigmoid":
		Z , linear_cache = linear_forward(A_prev , W  , b)
		A , activation_cache = sigmoid(Z)

	if activation == "relu":
		Z , linear_cache = linear_forward(A_prev , W , b)
		A , activation_cache = relu(Z)

	cache = (linear_cache , activation_cache)
	return A , cache


def forward_prop(X , parameters):
	 

	L = len(parameters)//2
	caches = []
	A = X 

	for l in range(1 , L):
		A_prev = A
		A , cache = linear_forward_deep(A_prev , parameters["W"+str(l)] , parameters["b"+str(l)] , "relu")
		caches.append(cache)

	AL , cache  = linear_forward_deep(A , parameters["W"+str(L)] , parameters["b"+str(L)] , "sigmoid")
	caches.append(cache)

	return AL , caches

def compute_cost(AL , Y):
	m = Y.shape[1]

	cost = (1/m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))

	#cost = (-1/m) * np.dot(Y , np.log(AL).T)
	cost = np.squeeze(cost)
	return cost

def linear_backward(dZ , cache):
	
	A_prev ,  W , b = cache
	m = A_prev.shape[1]
	dW = (1/m) * np.dot(dZ , A_prev.T)
	db = (1/m) * np.sum(dZ , axis = 1 , keepdims = True)
	dA_prev = np.dot(W.T , dZ)
	return dA_prev , dW , db

def linear_activation_backward(dA , cache , activation):
	linear_cache , activation_cache = cache
	if activation == "relu":
		dZ = backward_relu(dA , activation_cache)
		dA_prev , dW ,db = linear_backward(dZ , linear_cache)

	if activation == "sigmoid":
		dZ = backward_sigmoid(dA , activation_cache)
		dA_prev , dW , db = linear_backward(dZ , linear_cache)

	return dA_prev , dW , db

def backward_prop(AL , Y , caches):

	grads = {}
	L = len(caches)  # no of layers
	m = AL.shape[1]
	Y = Y.reshape(AL.shape)


	
	dAL = - (np.divide(Y , AL) - np.divide(1-Y , 1-AL))

	current_cache = caches[-1]
	grads["dA"+str(L-1)] , grads["dW"+str(L)]  , grads["db"+str(L)] = linear_activation_backward(dAL , current_cache , "sigmoid")

	#loop from l = L-2 to 0
	for l in reversed(range(L-1)):
		current_cache = caches[l]
		dA_prev_temp , dW_temp , db_temp = linear_activation_backward(grads["dA"+str(l+1)] , current_cache , "relu")
		grads["dA"+str(l)] = dA_prev_temp
		grads["dW"+str(l+1)] = dW_temp
		grads["db"+str(l+1)] = db_temp

	return grads

def update_parameters(parameters , grads , learning_rate):
	L = len(parameters)//2
	for l in range(L):
		parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate * grads["dW"+str(l+1)]
		parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate * grads["db"+str(l+1)]

	return parameters

def predict(X ,  parameters):

	AL , cache = forward_prop(X , parameters)
	Y_Prediction = np.zeros([1 , X.shape[1]])
	m = X.shape[1]

	for i in range(0 , AL.shape[1]):
		if AL[0,i] > 0.5:
			Y_Prediction[0,i] = 1
		else:
			Y_Prediction[0 , i] = 0
	#print("Accuracy: "  + str(np.sum((Y_Prediction == y)/m)))

	return Y_Prediction

	#print("Accuracy"+str(np.sum((Y_Prediction==Y)/m)))

def accuracy(X , y,  parameters):

	AL , cache = forward_prop(X , parameters)
	Y_Prediction = np.zeros([1 , X.shape[1]])
	m = X.shape[1]

	for i in range(0 , AL.shape[1]):
		if AL[0,i] > 0.5:
			Y_Prediction[0,i] = 1
		else:
			Y_Prediction[0 , i] = 0
	print("Accuracy: "  + str(np.sum((Y_Prediction == y)/m)))

	return Y_Prediction
