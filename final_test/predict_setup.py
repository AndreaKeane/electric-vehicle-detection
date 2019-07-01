#!/usr/bin/python3
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import pandas as pd
import pickle
import utilities.utilities as utils

def a_features(raw_data): 
	'''Builds required features for Part A Model'''

	# Difference from preceding interval
	feat_diff = raw_data.diff(axis=1)

	# Identify various possible features/descriptors
	features = pd.DataFrame(index=raw_data.index)

	features['avg_pwr'] = raw_data.mean(axis=1)
	features['avg_pwr^2'] = features['avg_pwr']**2
	features['avg_pwr^3'] = features['avg_pwr']**3

	features['median_pwr'] = raw_data.median(axis=1)
	features['median_pwr^2'] = features['median_pwr']**2
	features['median_pwr^3'] = features['median_pwr']**3

	features['min_pwr'] = raw_data.min(axis=1)
	features['min_pwr^2'] = raw_data.min(axis=1)**2
	features['min_pwr^3'] = raw_data.min(axis=1)**3

	features['max_pwr'] = raw_data.max(axis=1)
	features['max_pwr^2'] = features['max_pwr']**2
	features['max_pwr^3'] = features['max_pwr']**3

	features['diff_max'] = feat_diff.max(axis=1)
	features['diff_max^2'] = features['diff_max']**2
	features['diff_max^3'] = features['diff_max']**3

	features['<3_avg'] = raw_data[raw_data < 3].mean(axis=1)   
	features['>3_avg'] = raw_data[raw_data > 3].mean(axis=1)  
	features['pct_pwr<2'] = raw_data[raw_data < 2].count(axis=1)/raw_data.shape[1]   
	features['pct_pwr<3'] = raw_data[raw_data < 3].count(axis=1)/raw_data.shape[1]   

	features.dropna(axis=1, inplace=True)

	return features


def b_features(raw_data): 
	'''Builds required features for Part B Model'''
	
	# Difference from preceding interval
	feat_diff = raw_data.diff(axis=1)

	# 8-hour average
	feat_8h_avg = raw_data.rolling(window=(8*2), axis=1).mean()

	# 24-hour average
	feat_24h_avg = raw_data.rolling(window=(24*2), axis=1).mean()

	# 24-hour min
	feat_24h_min = raw_data.rolling(window=(24*2), axis=1).min()

	# 24-hour max
	feat_24h_max = raw_data.rolling(window=(24*2), axis=1).max()

	# 72-hour average
	feat_72h_avg = raw_data.rolling(window=(72*2), axis=1).mean()

	features = pd.DataFrame(index=raw_data.stack().index)
	for i, index in enumerate(features.index.tolist()): 
	    house_id, interval = index
	    features.at[index, 'value'] = raw_data[interval].loc[house_id]
	    features.at[index, 'diff'] = feat_diff[interval].loc[house_id]
	    features.at[index, 'h8_avg'] = feat_8h_avg[interval].loc[house_id]
	    features.at[index, 'h24_avg'] = feat_24h_avg[interval].loc[house_id]
	    features.at[index, 'h24_min'] = feat_24h_min[interval].loc[house_id]
	    features.at[index, 'h24_max'] = feat_24h_max[interval].loc[house_id]
	    features.at[index, 'h72_avg'] = feat_72h_avg[interval].loc[house_id]

	# Create combinations of features
	features['diff_2'] = features['diff']**2
	features['diff_3'] = features['diff']**3
	features['diff_5'] = features['diff']**5

	# Drop NA-containing rows 
	before = set(features.index.tolist())
	features = features.dropna(axis=0)
	after = set(features.index.tolist())
	removed_id = before - after
	print("{} Rows were dropped for NA reasons.".format(len(removed_id)))

	return features

def predict_a(test):
	''' '''	

	pickle_path = Path('/Users/andreakeane/Documents/DataScience/GridCure_Problems/pickles/')

	# Open classifier, scaler
	a_lr = utils.get_pickle(pickle_path / "a_logreg.pkl")
	a_scaler = utils.get_pickle(pickle_path / "a_scaler.pkl")
	
	# Setup data
	X = a_features(test)
	X_scaled = pd.DataFrame(a_scaler.transform(X), index=X.index, columns=X.columns)

	# Make predictions
	a_probs = a_lr.predict_proba(X_scaled)[:,1]
	a_predicts = a_lr.predict(X_scaled)

	# Convert to df
	df_out = pd.DataFrame(data=a_predicts, index=X_scaled.index, columns=['predict'])

	return df_out

def predict_b(test):
	''' '''
	pickle_path = Path('/Users/andreakeane/Documents/DataScience/GridCure_Problems/pickles/')

	# Open classifier, scaler
	classifier = utils.get_pickle(pickle_path / "b_knn.pkl")
	scaler = utils.get_pickle(pickle_path / "b_scaler.pkl")

	# Setup data
	X = b_features(test)
	X_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
	
	# Make predictions
	b_probs = classifier.predict_proba(X_scaled)[:,1]
	b_predicts = classifier.predict(X_scaled)

	# Convert to df
	df_out = pd.DataFrame(data=b_probs, index=X_scaled.index)

	return df_out.unstack()

