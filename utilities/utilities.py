#!/usr/bin/python3
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split 
from pathlib import Path
import pandas as pd
import pickle


def generate_test_results(X_test, y_test, model):
	'''Joins X_test, y_test, y_pred and confusion results.'''
	results = X_test.join(y_test.rename('label_true'))

	# Predict bool classification
	y_pred = model.predict(X_test)
	y_pred = pd.DataFrame(y_pred, 
						index=X_test.index, 
						columns=['label_pred'])

	# Probability for each testing classification
	probs = model.predict_proba(X_test)[:,1]
	probs = pd.DataFrame(probs, 
						index=X_test.index, 
						columns=['label_prob'])

	results = results.join(y_pred)
	results = results.join(probs)
	
	return results


def scale_split_data(X, y):
	'''Scale and split the data. Return split data'''

	# Scale X-data between -1 and 1
	scaler = StandardScaler().fit(X)                                    
	X_scaled = pd.DataFrame(scaler.transform(X), 
							index=X.index, 
                        	columns=X.columns)

	# Split data into training and testing
	X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)

	return X_train, X_test, y_train, y_test



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
	'''Identifies House_IDs to be considered outliers'''
	# Summary stats for each training sample
	stats = pd.DataFrame(index=train_df.index)
	stats['max'] = pd.DataFrame(train_df.max(axis=1))
	# NOTE: Reserving these for future use in outlier classification
	# stats['sum'] = pd.DataFrame(train_df.sum(axis=1))
	# stats['min'] = pd.DataFrame(train_df.min(axis=1))
	# stats['avg'] = pd.DataFrame(train_df.mean(axis=1))

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
	'''Pickles raw EV_files data as Pandas dataframes.'''
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

def pickle_house_classes():
	'''Pickles lists segregating the houses with and without EVs'''
	# Path Setup
	pickle_path = Path('/Users/andreakeane/Documents/DataScience/GridCure_Problems/pickles/')
	labels = pd.read_pickle(pickle_path / "labels.pkl")

	## Determine houses with and without EVs
	temp = pd.DataFrame(index=labels.index)
	temp['sum'] = labels.sum(axis=1)
	temp['bool'] = np.where(temp['sum'] > 0, True, False)

	houses_wEV = temp.index[temp['bool'] == True].tolist()
	houses_woEV = temp.index[temp['bool'] == False].tolist()

	print("{} Houses with EVs.".format(len(houses_wEV)))
	print("{} Houses without EVs.".format(len(houses_woEV)))

	# Pickle the data for reference elsewhere
	# Lists, so we can't use Pandas pickling method
	make_pickle(houses_wEV, pickle_path / "houses_wEV.pkl")
	make_pickle(houses_woEV, pickle_path / "houses_woEV.pkl")


def make_pickle(cucumber, pickle_jar):
	'''
	Pickles non-Pandas objects.
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




