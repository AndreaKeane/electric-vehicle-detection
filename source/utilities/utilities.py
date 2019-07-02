#!/usr/bin/python3
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from pathlib import Path
import pandas as pd
import pickle


def logreg_evaluation(model, X_train, fig_label, fig_path):
	'''
	Performs evaluation specific to the LogisticRegression Model
	'''

	# Evaluate Intercept
	intercept = model.intercept_[0]
	intercept_odds = np.exp(intercept)
	intercept_prob = intercept_odds / (1 + intercept_odds)

	print("INTERCEPT\n{:.1f}% probability of house having an EV".format(intercept_prob*100))

	# Evaluate coefficients
	analysis = pd.DataFrame(index=X_train.columns)
	analysis['logodds'] = np.array(model.coef_[0])
	analysis['odds'] = np.exp(analysis['logodds'])
	analysis['probs'] = analysis.odds / (1 + analysis.odds)

	# Display as heatmap
	f, ax = plt.subplots(figsize=(2, 6))
	sns.heatmap(analysis.drop(['logodds', 'odds'], axis=1), 
				annot=True, robust=True, 
				xticklabels=["Coefficient\nProbabilities"], 
				linewidths=.5
			)
	f.savefig(fig_path / (fig_label + "_coef_heatmap.png"), 
			dpi=400,  bbox_inches='tight')


def model_evaluation(model, X_train, y_train, X_test, y_test, fig_label, fig_path):
	'''
	model - classifier object 
	[train, test] data
	fig_label - string descriptor to differentiate saved figs

	Requires model to have `predict` and `predict_proba` methods 
	'''
	from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, balanced_accuracy_score

	y_true = y_test
	y_scores = model.predict(X_test)  
	y_probs = model.predict_proba(X_test)[:,1]

	# ROC
	auc = roc_auc_score(y_true, y_probs)
	print("ROC AUC Score: {:.3f}".format(auc))
	fpr, tpr, thresholds_roc = roc_curve(y_true, y_probs)

	# Precision-Recall (PR) Curve
	ap = average_precision_score(y_true, y_probs)  
	print("Average Precision Score: {:.3f}".format(ap))
	p, r, thresholds_pr = precision_recall_curve(y_true, y_probs)

	# Balanced Accuracy
	balanced_acc = balanced_accuracy_score(y_true, y_scores)
	print("Balanced Accuracy Score: {:.3f}".format(balanced_acc))

	# Generate ROC Curve
	sns.lineplot(x=fpr, y=tpr, markers='.')
	sns.lineplot(x=[0,1], y=[0,1])
	plt.legend(('ROC Curve', 'Random Boundary'), frameon=True) 
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic (ROC) Curve')

	plt.savefig(fig_path / (fig_label + "_ROC.png"), dpi=400)
	plt.show()

	# Generate PR Curve
	t = np.append(thresholds_pr, 1) 
	if p.shape[0] != r.shape[0] != thresholds.shape[0]: 
		print("Error. Incorrect Shapes")
		print(p.shape[0], r.shape[0], t.shape[0])

	# Plot option #1
	sns.lineplot(t, p) 
	sns.lineplot(t, r)
	plt.legend(('Precision', 'Recall'), frameon=True) 
	plt.xlabel('Threshold') 
	plt.ylabel('Proportion')
	plt.title('Precision-Recall (PR) Curve')
	plt.show()

	# Plot Option #2
	min_p = min(p)
	sns.lineplot(x=r, y=p)
	# sns.lineplot(x=[0,1], y=[min_p,min_p])
	plt.legend(('PR Curve'), frameon=True) 
	plt.ylim(bottom=0)   
	plt.xlabel('Recall') 
	plt.ylabel('Precision')
	plt.title('Precision-Recall (PR) Curve')

	plt.savefig(fig_path / (fig_label + "_PR.png"), dpi=400)
	plt.show()

	# Score Summary
	# Prints as CSV for easy reformatting
	print("Training Score, {:.3f}".format(model.score(X_train, y_train)))
	print("Testing Score, {:.3f}".format(model.score(X_test, y_test)))
	print("ROC AUC, {:.3f}".format(auc))
	print("Average Precision Score, {:.3f}".format(ap))
	print("Balanced Accuracy Score, {:.3f}".format(balanced_acc))


def scale_split_data(X, y, scaler_label=None):
	'''
	Scale and split the data. 
	Return split data.
	scaler_label - String, saves scaler object with this label
	'''
	pickle_path = Path('../../pickles')
	

	# Scale X-data between -1 and 1
	scaler = StandardScaler().fit(X)                                    
	X_scaled = pd.DataFrame(scaler.transform(X), 
							index=X.index, 
                        	columns=X.columns)

	# Split data into training and testing
	X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)

	# Pickle the Scaler so we can apply it on the test data
	if scaler_label: 
		make_pickle(scaler, pickle_path / (scaler_label + "_scaler.pkl"))

	return X_train, X_test, y_train, y_test


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


if __name__ == "__main__": 
	print("This script is not intended to be a standalone executable.")
	sys.exit()

