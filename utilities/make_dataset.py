#!/usr/bin/python3
import numpy as np
import pandas as pd
from pathlib import Path
import utilities as utils

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
	utils.make_pickle(houses_wEV, pickle_path / "houses_wEV.pkl")
	utils.make_pickle(houses_woEV, pickle_path / "houses_woEV.pkl")

if __name__ == "__main__": 
    pickle_raw_data()
    pickle_house_classes()




