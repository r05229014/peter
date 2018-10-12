import sys, argparse, os, time, pickle
import numpy as np

from sklearn.linear_model import LinearRegression
from Preprocessing import load_data, Preprocessing_Linear, Preprocessing_DNN
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint


parser = argparse.ArgumentParser(description='Peter data hahahahha')
parser.add_argument('model_name')
parser.add_argument('action', choices=['train', 'test'])
parser.add_argument('model_use', choices=['Linear', 'DNN'])

# training argument
parser.add_argument('--batch_size', default=512, type=float)
parser.add_argument('--nb_epoch', default=20, type=int)
parser.add_argument('--test_ratio', default=0.2, type=float)
parser.add_argument('--resolution', '--res', default='d32', type=str)

# model parameter
parser.add_argument('--loss_function', default='mean_squared_error')
parser.add_argument('-hid_siz', '--hidden_size', default=512, type=int)
parser.add_argument('--dropout_rate', default=0.3, type=float)
parser.add_argument('-lr','--learning_rate', default=0.001,type=float)
parser.add_argument('--threshold', default=0.1,type=float)

# put model in the same directory
parser.add_argument('--model_save_path', default='../model/temp/',)
parser.add_argument('--history_save_path', default='../history/temp/',)
parser.add_argument('--load_model_path', default = None)
#parser.add_argument('--save_dir', default = 'model/')
args = parser.parse_args()

tStart = time.time()

path = '../data/pickle/'
res = args.resolution


def DNN():
	print("Build model!!")
	model = Sequential()
	model.add(Dense(args.hidden_size, activation = 'relu', input_shape=(4,)))
	model.add(Dense(args.hidden_size, activation = 'relu'))
	model.add(Dense(1, activation='relu'))

	# optimize 
	adam = Adam()

	print('Compiling model...')
	model.compile(loss=args.loss_function, optimizer=adam)
	print(model.summary())
	
	return model


def main():
	X_train, X_test, y_train, y_test = load_data(path, res, args.test_ratio)
	X_train, X_test, y_train, y_test = Preprocessing_DNN(X_train, X_test, y_train, y_test)

	# Training
	if args.action == 'train':
		if args.model_use == 'Linear':
			model = LinearRegression()
			model.fit(X_train, y_train)
			with open(args.model_save_path + 'Linear_model.pik', 'wb') as f:
				pickle.dump(model, f)

		elif args.model_use == 'DNN':
			model = DNN()
			filepath = args.model_save_path + "/weights-improvement-{epoch:03d}-{loss:.3e}.hdf5"
			checkpoint = ModelCheckpoint(filepath, monitor='val_loss',save_best_only=False, period=1)
			history = model.fit(X_train, y_train, validation_data=(X_test, y_test), shuffle=True, callbacks=[checkpoint])
		
			# save history
			with open(args.history_save_path + args.model_name + '.pkl', 'wb') as f:
				pickle.dump(history.history, f)

	# Testing
	elif args.action == 'test':
		if args.model_use == 'Linear':
			with open(args.load_model_path, 'rb') as f:
				model = pickle.load(f)

		elif args.model_use == 'DNN':
			model = load_model(args.load_model_path)

		model.predict(X_test)

if __name__ == '__main__':
	main()

