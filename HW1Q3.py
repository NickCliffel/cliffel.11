import numpy as np
import pandas as pd

def DataSet() -> dict:
	x = np.array([
		[0,0,0,0],
		[0,0,1,0],
		[0,0,0,1],
		[0,0,1,1],
		[0,1,0,0],
		[0,1,1,0],
		[0,1,0,1],
		[0,1,1,1],
		[1,0,0,0],
		[1,0,1,0],
		[1,0,0,1],
		[1,0,1,1],
		[1,1,0,0],
		[1,1,1,0],
		[1,1,0,1],
		[1,1,1,1]
	])
	y = np.array([
		[0,0],
		[1,0],
		[0,1],
		[1,1],
		[0,1],
		[1,1],
		[1,0],
		[0,0],
		[1,0],
		[0,0],
		[1,1],
		[0,1],
		[1,1],
		[0,1],
		[0,0],
		[1,0]
	])
	return {'x':x,'y':y}

def Act(v) -> int:
	return 1 if v >= 0 else 0

def ComputeLayer(w:np.ndarray, a:np.ndarray) -> np.ndarray:
	# adding bias to x column
	a = np.column_stack((np.ones((a.shape[0],)), a))
	vecFunc = np.vectorize(Act)
	return vecFunc(w @ a.T)

def Network() -> [np.ndarray]:
	layer1 = np.array([
		[-1,0,-1,0,1], # part of xor for y2
		[-1,0,1,0,-1], # part of xor for y2
		[-2,0,1,0,1], # and of x2 and x4
		[-1,-1,0,1,0], # -x1 and x3
		[-1,1,0,-1,0] # x1 and -x3
	])
	layer2 = np.array([
		[-1,1,1,0,0,0], # this is the y2 value
		[-1,0,0,0,1,1], # xor of x1 and x3
		[-1,0,0,1,0,0], # passing on and of x2 and x4
	])
	layer3 = np.array([
		[-1,1,0,0], # passing on y2 value
		[-1,0,1,-1],
		[-1,0,-1,1]
	])
	layer4 = np.array([
		[-1,0,1,1],
		[-1,1,0,0]
	])
	return [layer1,layer2,layer3,layer4]

def Activation(v:np.ndarray) -> np.ndarray:
	return np.array([1 if val >= 0 else 0 for val in v])

def Calculate(x:np.ndarray, w:[np.ndarray]):
	a = x
	for i in range(len(w)):
		a = ComputeLayer(w[i], a).T
	#	print(a)
	return a

def main():
	dataDict = DataSet()
	x = dataDict['x']
	y = dataDict['y']
	#toPredict = x[2,:].reshape(1,4)
	#print(toPredict.shape)
	#Calculate(toPredict, Network())
	calced = Calculate(x.copy(), Network())
	print('input -> expected -> calculated -> correct')
	for i in range(x.shape[0]):
		correct = True if y[i,0] == calced[i,0] and y[i,1] == calced[i,1] else False
		print(f'{x[i]} -> {y[i]} -> {calced[i]} -> {correct}')

if __name__ == '__main__': main()
