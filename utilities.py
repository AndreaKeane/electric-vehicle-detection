#!/usr/bin/python3
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split 
from pathlib import Path
import pandas as pd


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


def get_raw_data():
	''' '''
	# File Setup
	# from pathlib import Path
	data_folder = Path('/Users/andreakeane/Documents/DataScience/GridCure_Problems/EV_files/')

	# Test Data
	test = pd.read_csv(data_folder / "EV_test.csv", index_col='House ID')

	# Train Data
	train = pd.read_csv(data_folder / "EV_train.csv", index_col='House ID')

	# Train Labels
	labels = pd.read_csv(data_folder / "EV_train_labels.csv", index_col='House ID')

	print("Data Frame Shapes")
	print("Test: {}\n"
	      "Train: {}\n"
	      "Labels: {}".format(test.shape, train.shape, labels.shape))

	return test, train, labels