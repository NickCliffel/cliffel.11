import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Plot(xArr:np.ndarray, yArr:np.ndarray):
	plt.scatter(xArr[:,0], xArr[:,1], c=yArr)
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.show()

def Act(val:float) -> int:
	return 1 if val >= 0 else -1

def Test(x:np.ndarray, w:np.ndarray) -> np.ndarray:
	result = []
	x = np.column_stack((np.ones((x.shape[0],)),x))
	for i in range(x.shape[0]):
		result.append(Act(w @ x[i,:].T))
	return np.array(result)

def plr(x:np.ndarray, w:np.ndarray, d:np.ndarray, n:float, L:int) -> np.ndarray:
	# L -> number of iterations, w -> weights, n -> learning rate, d -> desired outputs
	# adding bias to data
	x = np.column_stack((np.ones((x.shape[0],)), x))
	
	# going through the number of iterations
	for i in range(L):
		pos = i % x.shape[0]
		updateVal = n * (d[pos] - Act(w @ x[pos,:].T)) * x[pos,:]
		w = w + updateVal

	return w

def main():
	# defining data
	xArr = np.array([[0,0],[0,1],[1,0],[1,1]])
	yArr = np.array([1,1,-1,-1])

	xArr2 = np.array([[0,0],[0,1],[1,0],[1,1]])
	yArr2 = np.array([-1,1,1,-1])

	
	# plotting data
	#Plot(xArr, yArr)

	# initiliazing weights
	w = np.zeros((xArr.shape[1] + 1,))

	# applying perceptron learning rule
	w = plr(x=xArr, w=w, d=yArr, n=0.5, L=4)

	print(f'weights: {w}')
	print(f'expected: {yArr}')
	print(f'result: {Test(x=xArr, w=w)}')

	w2 = np.zeros((xArr2.shape[1] + 1,))
	w2 = plr(x=xArr2, w=w2, d=yArr2, n=0.5, L=4)

	print()
	print(f'weights: {w2}')
	print(f'expected: {yArr2}')
	print(f'result: {Test(x=xArr2, w=w2)}')

if __name__ == '__main__': main()
