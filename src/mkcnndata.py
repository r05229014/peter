import sys, argparse, os, time, pickle
import numpy as np


def load_data(path, res):
	print('Loading data from %s' %path+res)
	
	with open(path + res + '.pkl', 'rb') as f:
		case = pickle.load(f)
	for key, value in case.items():
		case[key] = np.squeeze(value)
		case[key] = value.reshape(value.shape[0], value.shape[2], value.shape[3], value.shape[1])
		print(key, case[key].shape)
			
	X = np.concatenate((case['u'], case['v'], case['th'], case['qv']), axis=-1)
	y = case['sigma']
	
	return X,y
	# shuffle
	#indices = np.arange(X.shape[0])
	#nb_test_samples = int(TEST_SPLIT * X.shape[0])
	#np.random.shuffle(indices)
	#X = X[indices]
	#y = y[indices]

	#X_train = X[nb_test_samples:]
	#X_test = X[0:nb_test_samples]
	#y_train = y[nb_test_samples:]
	#y_test = y[0:nb_test_samples]

	#print('X_train shape is : ', X_train.shape)
	#print('X_test shape is : ', X_test.shape)
	#print('y_train shape is : ', y_train.shape)
	#print('y_test shape is : ', y_test.shape)
	#print('\n')


def pool_reflect(array):
	new = np.zeros((array.shape[0], array.shape[1]+2, array.shape[2]+2, array.shape[3]))
	for sample in range(array.shape[0]):
		for feature in range(array.shape[3]):
			#print("sample %s" %sample)
			tmp = array[sample,:,:,feature]
			tmp_ = np.pad(tmp, 1, 'wrap')
			new[sample,:,:,feature] = tmp_
	return new


def cnn_type_x(arr):
	out = np.zeros((arr.shape[0]*(arr.shape[1]-2)*(arr.shape[2]-2), 3, 3, arr.shape[3]))

	count = 0
	for s in range(arr.shape[0]):
		for x in range(0,arr.shape[1]-2):
			for y in range(0,arr.shape[2]-2):
				out[count] = arr[s, x:x+3, y:y+3, :]
				
				count += 1
	print(count)
	print('X sahape : ',out.shape)
	return out


def cnn_type_y(arr):
	out = np.zeros((arr.shape[0]*(arr.shape[1])*(arr.shape[2]), 1, 1, arr.shape[3]))

	count = 0
	for s in range(arr.shape[0]):
		for x in range(0,arr.shape[1]):
			for y in range(0,arr.shape[2]):
				out[count] = arr[s, x, y, :]
				
				count += 1
	print(count)
	print('y shape : ',out.shape)
	return out

if __name__ == '__main__':
	path = '../data/pickle/'
	res = 'd32'
	X,y = load_data(path, res)
	X = pool_reflect(X)
	X = cnn_type_x(X)
	y = cnn_type_y(y)

