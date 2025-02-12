import numpy as np
import pandas as pd

def Forward(w:list, x:np.ndarray):
	pass

def Act(v) -> int:
	return 1 if v >= 0 else -1

def ComputeLayer(w:np.ndarray, a:np.ndarray) -> np.ndarray:
	# adding bias to x column
	a = np.column_stack((np.ones((a.shape[0],)), a))
	vecFunc = np.vectorize(Act)
	print(f'a: {a}')
	print(f'v: {w @ a.T}')
	print()
	return vecFunc(w @ a.T)

def Forward(x:np.ndarray, w:[np.ndarray]) -> np.ndarray:
	a = x
	for i in range(len(w)):
		a = ComputeLayer(w[i], a).T
		print(a)
	return a

def Compare(pred:np.ndarray, y:np.ndarray, x:np.ndarray):
	if pred.shape[0] != y.shape[0] or pred.shape[1] != y.shape[1]:
		raise Exception('PRED AND Y SHAPES DO NOT MATCH')
	print('x -> actual -> predicted -> correct')
	for i in range(x.shape[0]):
		correct = pred[i,0] == y[i,0] and pred[i,1] == y[i,1] and pred[i,2] == y[i,2]
		print(f'{x[i,:]} -> {y[i,:]} -> {pred[i,:]} -> {correct}')

def main():
	# setting up network
	w1 = np.array([
		[4, 1, 0], # x1 >= -4
		[-2, -1, 0], # x1 <= -2
		[2, 0, -1], # x2 <= 2
		[0, 0, 1], # x2 >= 0
		[-3, 1, 0], # x1 >= 3
		[7, -1, 0], # x1 <= 7
		[3.75, -0.25, -1], # line for top class 2
		[5, 0, -1], # x2 <= 5
		[-3, 0, 1], # x2 >= 3
		[3, 2/3, -1], # line for left of class 3
		[-5/3, -2/3, 1] # line for right of class 3
	])
	w2 = np.array([
		[-4, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], # class 1
		[-4, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], # class 2
		[-4, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1] # class 3
	])
	w = [w1,w2]
	
	# creating test data
	x = np.array([
		[-4, 1], #1
		[-2, 1], #1
		[-3, 1], #1
		[-3, 0], #1
		[-3, 2], #1
		[-5, 1], #4
		[1, 1], #4
		[-3, -1], #4
		[-3, 3], #4
		[3, 1], #2
		[7, 1], #2
		[5, 0], #2
		[5, 2.5], #2
		[5, 1], #2
		[5, 2.75], #4
		[8, 1], #4
		[2, 1], #4
		[5, -1], #4
		[1, 3], #3
		[4, 5], #3
		[1, 2], #4
		[4, 6], #4
		[3, 11/3], #3
		[4, 13/3], #3
		[5, 5], #3
		[3, 12/3], #3
		[1, 11/3], #3
		[2, 13/3], #3
		[3, 5], #3
		[2, 13/3], #3
		[3, 4], #3
		[1, 4], #4
		[3, 3.5] #4
	])
	y = np.array([
		[1,-1,-1],
		[1,-1,-1],
		[1,-1,-1],
		[1,-1,-1],
		[1,-1,-1],
		[-1,-1,-1],
		[-1,-1,-1],
		[-1,-1,-1],
		[-1,-1,-1],
		[-1,1,-1],
		[-1,1,-1],
		[-1,1,-1],
		[-1,1,-1],
		[-1,1,-1],
		[-1,-1,-1],
		[-1,-1,-1],
		[-1,-1,-1],
		[-1,-1,-1],
		[-1,-1,1],
		[-1,-1,1],
		[-1,-1,-1],
		[-1,-1,-1],
		[-1,-1,1],
		[-1,-1,1],
		[-1,-1,1],
		[-1,-1,1],
		[-1,-1,1],
		[-1,-1,1],
		[-1,-1,1],
		[-1,-1,1],
		[-1,-1,1],
		[-1,-1,-1],
		[-1,-1,-1]
	])

	xSmall = np.array([
		[3, 11/3]
	])

	ySmall = np.array([
		[-1,-1,1],
	])	

	pred = Forward(x=x, w=w)
	print(f'pred shape: {pred.shape}')
	Compare(pred, y, x)

if __name__ == '__main__': main()
