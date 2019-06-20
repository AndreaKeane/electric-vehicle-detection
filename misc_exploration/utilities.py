#!/usr/bin/python3
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split 
from pathlib import Path
import pandas as pd
import pickle


def logreg_model(X, y):
	'''Performs redundant logistic regression modelling'''

	# Scale X-data between -1 and 1
	scaler = StandardScaler().fit(X)                                    
	X_scaled = scaler.transform(X)

	# Split data into training and testing
	X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)

	# Train logistic regression model
	logreg = LogisticRegression(solver='liblinear', random_state=0)
	logreg.fit(X_train, y_train)
	y_pred = logreg.predict(X_test)
	print('Accuracy of classifier on test set: {:.3f}'.format(logreg.score(X_test, y_test)))

	return logreg


def classify_outliers(train_df):
	''' '''
	# Summary stats for each training sample
	stats = pd.DataFrame(index=train_df.index)
	stats['sum'] = pd.DataFrame(train_df.sum(axis=1))
	stats['min'] = pd.DataFrame(train_df.min(axis=1))
	stats['max'] = pd.DataFrame(train_df.max(axis=1))
	stats['avg'] = pd.DataFrame(train_df.mean(axis=1))

	# Use stats to classify outliers
	std = stats['max'].std()
	mean = stats['max'].mean()
	threshold = mean + (2 * std)
	outliers = stats.loc[stats['max'] > threshold]

	outlier_ct = outliers.shape[0]
	print("Number of Outliers: {}".format(outlier_ct))
	print("Percent Removed: {}%\n".format(round(outlier_ct/train_df.shape[0]*100, 2)))

	return outliers


def pickle_raw_data():
	''' '''
	# Path Setup
	data_path = Path('/Users/andreakeane/Documents/DataScience/GridCure_Problems/EV_files/')
	pickle_path = Path('/Users/andreakeane/Documents/DataScience/GridCure_Problems/pickles/')

	# Read data to DataFrames
	test = pd.read_csv(data_path / "EV_test.csv", index_col='House ID')
	train = pd.read_csv(data_path / "EV_train.csv", index_col='House ID')
	labels = pd.read_csv(data_path / "EV_train_labels.csv", index_col='House ID')

	print("Data Frame Shapes")
	print("Test: {}\n"
	      "Train: {}\n"
	      "Labels: {}".format(test.shape, train.shape, labels.shape))

	# Write out pickles
	test.to_pickle(pickle_path / "test.pkl")
	train.to_pickle(pickle_path / "train.pkl")
	labels.to_pickle(pickle_path / "labels.pkl")

def make_pickle(cucumber, pickle_jar):
	'''
	Pickles non-Pandas obejcts.
	cucumber - object to be pickled
	pickle_jar - filepath to save the pickle to
	'''
	with open(pickle_jar, 'wb') as pkl:
		pickle.dump(cucumber, pkl, -1)

def get_pickle(pickle_jar):
	'''
	Retrieves pickled non-Pandas objects
	pickle_jar - filepath to save the pickle to
	'''
	with open(pickle_jar, 'rb') as pkl:
		return pickle.load(pkl)




