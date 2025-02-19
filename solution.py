import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

"""
THE ANSWER TO THESE FILES CAN ALSO BE FOUND IN THE WriteUp.docx FILE. PICTURES OF THE PLOTS ARE ALSO
THERE. PICTURED PLOTS ARE FROM ITERATION 1 AND DO NOT REPRESENT NUMBERS LISTED BELOW ITERATION 2. FORGOT
TO SET SEED VALUE.

1:
learning rate -> epochs
0.05 -> 1,694,503 
0.10 -> 2,643,293
0.15 -> 2,499,787
0.20 -> 1,891,269
0.25 -> 268,376
0.30 -> 428,449
0.35 -> 902,085
0.40 -> 158,224
0.45 -> 990,513
0.50 -> 896,522
The number of epochs general decreases as it goes the learning rate goes from 0.05 to 0.40 with
0.40 being the learning rate with the lowest number of epochs required. Although there is some jumping
aroung. After 0.40 the number of epochs required increases for 0.45 and 0.50.

2:
learning rate -> epochs
0.05 -> 168,353
0.10 -> 377,053
0.15 -> 258,223
0.20 -> 35,549
0.25 -> 23,687
0.30 -> 21,146
0.35 -> 20,677
0.40 -> 27,021
0.45 -> 20,186
0.50 -> 17,077

"""

# generates datapoints and returns them
def GetData() -> dict:
	# defining inputs
	X = np.array([
		[0,0,0,0], #1
		[0,0,0,1], #2
		[0,0,1,0], #3
		[0,0,1,1], #4
		[0,1,0,0], #5
		[0,1,0,1], #6
		[0,1,1,0], #7
		[0,1,1,1], #8
		[1,0,0,0], #9
		[1,0,0,1], #10
		[1,0,1,0], #11
		[1,0,1,1], #12
		[1,1,0,0], #13
		[1,1,0,1], #14
		[1,1,1,0], #15
		[1,1,1,1] #16
	])
	
	# defining desired outocome
	Y = np.array([1 if np.sum(X[i,:]) % 2 == 1 else 0 for i in range(X.shape[0])])
	
	# returnign inputs and outputs
	return {'X':X, 'Y':Y}

# defines the network weights and biases
def DefineNetwork(inputs: int, hiddenLayerNeurons:int, outputLayerNeurons:int):
	# initializing weights for hidden layer
	w1 = np.random.uniform(-1, 1, (inputs, hiddenLayerNeurons))

	# initializing biases for hidden layer
	b1 = np.random.uniform(-1, 1, (1, hiddenLayerNeurons))

	# initializing weights for output layer
	w2 = np.random.uniform(-1, 1, (hiddenLayerNeurons, outputLayerNeurons))
	
	# initializing biases for output layer
	b2 = np.random.uniform(-1, 1, (1, outputLayerNeurons))
	
	# returnign lists of weights and biases
	return [w1, w2], [b1, b2]

# sigmoid going forward
def Sigmoid(x:np.ndarray):
	return 1 / (1 + np.exp(-x))

# sigmoid going backward
def SigmoidBackward(dA_prev, x:np.ndarray):
	return dA_prev * x * (1 - x)

# used to conduct forward pass
def Forward(w, b, x:np.ndarray) -> dict:
	# initializing cache for backward pass
	cache = {'inputs':[], 'w':[], 'z':[], 'a':[]}
	
	# setting activation to initial inputs for the first layer
	a = x

	# iterating through layers
	for i in range(len(w)):
		# adding inputs and weights to the cache
		cache['inputs'].append(a)
		cache['w'].append(w[i])

		# calculating linear value and appending it to cache
		z = np.dot(a, w[i]) + b[i]
		cache['z'].append(z)

		# calculating activation value and appending it to cache
		a = Sigmoid(z)
		cache['a'].append(a)

	# returning cache for backward pass
	return cache

# calculates initial cost derivate
def Cost(actual, predicted):
	return -1 * (float(actual) - float(predicted))

# used to conduct the backward pass
def Backward(cache:dict, y) -> dict:
	# initializing return values
	gradients = {'dW':[], 'dB':[]}

	# calculating initial cost
	dA_prev = Cost(y, cache['a'][len(cache['a']) - 1])
	
	# iterating backwards through layers
	for i in reversed(range(len(cache['a']))):
		# calculating linear error
		dZ = SigmoidBackward(dA_prev, cache['a'][i])
		
		# calculating weight gradient and adding it to gradients dictionary
		dW = cache['inputs'][i].T.dot(dZ)
		gradients['dW'].insert(0, dW)

		# calculating biases gradient and adding it to gradients dictionary
		dB = dZ
		gradients['dB'].insert(0, dB)

		# caclulating inputs gradient for next layer
		dA_prev = dZ.dot(cache['w'][i].T)
	
	# returning gradients
	return gradients

# used to calculate absolute error over all of the samples
def AbsoluteError(X:np.ndarray, Y:np.ndarray, w:[np.ndarray], b:[np.ndarray]):
	# initializing error to 0
	error = 0
	
	# iterating over all examples
	for i in range(X.shape[0]):
		# conducting forward pass
		cache = Forward(w=w, b=b, x=X[i,:].reshape((1, X.shape[1])))
		
		# adding absolute error for sample to error sum
		error += abs(Y[i] - cache['a'][len(cache['a']) - 1])
	
	# returning error value
	return error

# used to update weights and biases
def UpdateGradients(w, b, gradients, n:float, B:float, momentumW:[np.ndarray], momentumB:[np.ndarray]):
	# initializing structures to store weights, biases, and moementum
	wNew = []
	bNew = []
	momentumW_New = []
	momentumB_New = []

	# iterating through network layers
	for i in range(len(w)):
		# calculating momentum for W and B
		momentumW_Temp = (B * momentumW[i]) - (n * gradients['dW'][i])
		momentumB_Temp = (B * momentumB[i]) - (n * gradients['dB'][i])
		
		# adding momentum values
		momentumW_New.append(momentumW_Temp)
		momentumB_New.append(momentumB_Temp)		

		# updating weights and biases
		wNew.append(w[i] + momentumW_Temp)
		bNew.append(b[i] + momentumB_Temp)
	
	# returning new weights, biases, and corresponding momentum values
	return wNew, bNew, momentumW_New, momentumB_New

def Train(w:[np.ndarray], b:[np.ndarray], X:np.ndarray, Y:np.ndarray, n:float, B:float) -> int:
	# getting initial error
	error = AbsoluteError(X=X, Y=Y, w=w, b=b)
	
	# initializing momentum values
	momentumW = []
	momentumB = []
	for i in range(len(w)):
		momentumW.append(np.zeros_like(w[i]))
		momentumB.append(np.zeros_like(b[i]))

	# variable to count the number of epochs needed
	epochsCount = 1

	# training until the error is less than 0.05
	while error >= 0.05:
		# going though each example
		for i in range(X.shape[0]):
			# foward pass
			cache = Forward(w=w, b=b, x=X[i,:].reshape((1, X.shape[1])))
			
			# backward pass
			gradients = Backward(cache=cache, y=Y[i])
			
			# updating weights
			w, b, momentumW, momentumB = UpdateGradients(w=w, b=b,
				gradients=gradients, n=n, B=B, momentumW=momentumW,
				momentumB=momentumB)
		
		# calculating new error value and outputting it
		error = AbsoluteError(X=X, Y=Y, w=w, b=b)
		if epochsCount % 100 == 0:
			print(f'ERROR AFTER EPOCH {epochsCount}: {error}, LEARNING RATE: {n}, MOMENTUM: {B}')
		
		# increasing epochs count
		epochsCount += 1
	
	# returning epochs count
	return epochsCount

# used to plot learning rate versus epochs
def Plot(holder:dict, title:str):
	plt.scatter(holder['lr'], holder['epochs'], marker='o', linestyle='-', color='b')
	plt.xlabel('learning rate')
	plt.ylabel('epochs')
	plt.title(title)
	plt.show()

def main():
	# getting data
	data = GetData()

	# initializing weights
	w, b = DefineNetwork(data['X'].shape[1], 4, 1)

	lr = [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

	# training the network
	holder = {'lr':lr, 'epochs':[]}
	for i in range(len(lr)):
		epochs = Train(w=w, b=b, X=data['X'], Y=data['Y'], n=lr[i], B=0)
		holder['epochs'].append(epochs)

	# writing results to json file
	with open('learning_rate_vs_epochs.json', 'w') as file:
		json.dump(holder, file, indent=4)

	# plotting learning rates versus epochs
	Plot(holder=holder, title='LEARNING RATE VS EPOCHS')

	# training the network with momentum
	holderMomentum = {'lr':lr, 'epochs':[]}
	for i in range(len(lr)):
		epochs = Train(w=w, b=b, X=data['X'], Y=data['Y'], n=lr[i], B=0.9)
		holderMomentum['epochs'].append(epochs)

	# writing results to json file
	with open('learning_rate_vs_epochs_with_momentum.json', 'w') as file:
		json.dump(holderMomentum, file, indent=4)

	# plottig learning rates versus epochs with momentum
	Plot(holder=holderMomentum, title='LEARNING RATE WITH MOMENTUM VS EPOCHS')

if __name__ == '__main__': main()
