# model eval
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ast import literal_eval
from itertools import chain
from sklearn.metrics import (accuracy_score, fowlkes_mallows_score, precision_score,
                             recall_score, f1_score, roc_auc_score, average_precision_score,
                             log_loss, brier_score_loss, precision_recall_curve, auc,
	                         roc_curve, confusion_matrix)
from sklearn.metrics import (explained_variance_score, mean_absolute_error, mean_squared_error)
from scipy.stats import zscore
from .general import GET_NUMERIC_AND_NONNUMERIC
from gopfsrisk_toolbox.algorithms import FIT_CATBOOST_MODEL
import statsmodels.api as sm


# ======================================================================================================
# CLASSES
# ======================================================================================================
class Model_Eval_Class():
	# init
	def __init__(self, logger=None):
		self.logger = logger

	# define function for featuer importance
	def save_feature_importance(model, filename='./output/df_featimp.csv'):
		# get model features
		list_model_features = model.feature_names_
		# get importance
		list_feature_importance = list(model.feature_importances_)
		# put in df
		df_imp = pd.DataFrame({'Feature': list_model_features,
								'Importance': list_feature_importance})
		# sort descending
		df_imp.sort_values(by='Importance', ascending=False, inplace=True)
		# try saving
		try:
			# save
			df_imp.to_csv(filename, index=False)
		except FileNotFoundError:
			# make output directory
			os.mkdir('./output')
			# save it
			df_imp.to_csv(filename, index=False)
		# save df_imp to self
		self.df_imp = df_imp
		# if logger
		if self.logger:
			self.logger.warning(f'Feature Importance saved to {filename}')

	# define function for splitting into x and y 
	def x_y_split(df_train, df_valid, df_test, str_targetname='TARGET__app'):
		# train
		self.y_train = df_train[str_targetname]
		del df_train[str_targetname]
		# valid
		self.y_valid = df_valid[str_targetname]
		del df_valid[str_targetname]
		# test
		self.y_test = df_test[str_targetname]
		del df_test[str_targetname]
		# if logging
		if self.logger:
			self.logger.warning('Train, valid, and test dfs split into X and y')
		# print message for conosle
		print('Train, valid, and test dfs split into X and y')
		# return
		return self

	# define function for PR Curve
	def pr_curve(y_true, y_hat_prob, y_hat_class, tpl_figsize=(10,10), filename='./output/plt_prcurve.png'):
		# get precision rate and recall rate
		precision_r, recall_r, thresholds = precision_recall_curve(y_true, y_hat_prob)
		# get f1
		flt_f1 = f1_score(y_true, y_hat_class)
		# get auc
		flt_auc = auc(recall_r, precision_r)
		# get the value of chance
		flt_chance = np.sum(y_true) / len(y_true)
		# create ax
		fig, ax = plt.subplots(figsize=tpl_figsize)
		# title
		ax.set_title(f'PR Curve: F1 = {flt_f1:0.3f}; AUC = {flt_auc:0.3f}')
		# xlabel
		ax.set_xlabel('Recall')
		# ylabel
		ax.set_ylabel('Precision')
		# pr curve
		ax.plot(recall_r, precision_r, label='PR Curve')
		# chance
		ax.plot(recall_r, [flt_chance for x in recall_r], color='red', linestyle=':', label='Chance')
		# legend
		ax.legend()
		# show plot
		plt.show()
		# save fig to self
		self.fig = fig
		# save fig
		plt.savefig(filename, bbox_inches='tight')
		# log for logging
		if self.logger:
			self.logger.warning(f'Precision-recall curve saved to {filename}')
		# print message for console
		print(f'Precision-recall curve saved to {filename}')
		# return
		return self

	# define function for ROC curves
	def roc_auc_curve(y_true, y_hat, tpl_figsize=(10,10), filename='./output/plt_roc_auc.png'):
		# get roc auc
		auc = roc_auc_score(y_true=y_true,
							y_score = y_hat)
		# get fpr, tpr
		fpr, tpr, thresholds = roc_curve(y_true=y_true,
										y_score=y_hat)
		# set up subplots
		fig, ax = plt.subplots(figsize=tpl_figsize)
		# set title
		ax.set_title('ROC Plot - (AUC: {0:0.2f})'.format(auc))
		# set x axis label
		ax.set_xlabel('False Positive Rate (Sensitivity)')
		# set y axis label 
		ax.set_ylabel('False Negative Rate (1 - Specificity)')
		# set x lim
		ax.set_xlim([0,1])
		# set y lim
		ax.set_ylim([0,1])
		# create curve
		ax.plot(fpr, tpr, label='Model')
		# plot diagonal red, dotted line
		ax.plot([0,1], [0,1], color='red', linestyle=':', label='Chance')
		# create legend
		ax.legend(loc='lower right')
		# fix overlap
		plt.tight_layout()
		# show plot
		plt.show()
		# save fig to self
		self.fig = fig
		# save fig
		plt.savefig(filename, bbox_inches='tight')
		# if using logger
		if self.logger:
			self.logger(f'ROC AUC curve saved in {filename}')
		# return fig
		return self

	# define function for residual plot
	def residual_plot(arr_yhat, ser_actual, filename='./output/residual_plt.png', tpl_figsize=(10,10)):
		# get residuals 
		ser_residuals = arr_yhat - ser_actual
		# get the norm
		norm = np.linalg.norm(ser_residuals)
		# normalize residuals
		ser_residuals = ser_residuals / norm
		# create axis
		fig, ax = plt.subplots(figsize=tpl_figsize)
		# title
		ax.set_title('Residual Plot (predicted - actual)')
		# distplot
		sns.distplot(ser_residuals, ax=ax)
		# save fig to self
		self.fig = fig
		# save fig
		plt.savefig(filename, bbox_inches='tight')
		# if logging, log it
		if self.logger:
			self.logger.warning(f'residual plot saved to {filename}')
		# return
		return self

	# define function for pd plots
	def pd_plot(model, X_train, y_train, list_cols, tpl_figsize=(15,10), dirname='./output/pd_plots', filename='./output/df_trends.csv'):
		# generate predictions first
		try:
			y_hat_train = model.predict_proba(X_train[model.feature_names_])[:,1]
		except AttributeError:
			y_hat_train = model.predict(X_train[model.feature_names_])
		# create dataframe
		X_train['predicted'] = y_hat_train
		X_train['actual'] = y_train 
		# create empty df
		df_empty = pd.DataFrame()
		# generate plots 
		for a, col in enumerate(list_cols):
			# print message
			print(f'Creating plot {a+1}/{len(list_cols)}')
			# group df
			X_train_grouped = X_train.groupby(by=col, as_index=False).agg({'predicted': 'mean',
																			'actual': 'mean'})
			# in case we have infinities or nan
			X_train_grouped = X_train_grouped[~X_train_grouped.isin([np.nan, np.inf, -np.inf]).any(1)]

			#sort
			X_train_grouped = X_train_grouped.sort_values(by=col, ascending=True)

			# make z score col name
			str_z_col = f'{col}_z'
			# get z score
			X_train_grouped[str_z_col] = zscore(X_train_grouped[col])
			# subset to only those with z >= 3 and <= -3 (remove outliers)
			X_train_grouped = X_train_grouped[(X_train_grouped[str_z_col] < 3) & (X_train_grouped[str_z_col] > -3)]

			# calculate trendlines
			# predicted
			z_pre = np.polyfit(X_train_grouped[col], X_train_grouped['predicted'], 1)
			n_pred = np.poly1d(z_pred)
			# actual
			z_act = np.polyfit(X_train_grouped[col], X_train_grouped['actual'], 1)
			p_act = np.poly1d(z_act)

			# create predicted array train 
			arr_trend_pred = p_pred(X_train_grouped[col])
			# create array for actual
			arr_trend_actual = p_act(X_train_grouped[col])

			# calculate run
			run_ = np.max(X_train_grouped[col]) - np.min(X_train_grouped[col])

			# calculate slope predicted
			flt_trend_pred = (arr_trend_pred[-1] - arr_trend_pred[0]) / run_
			# calculate slope actual
			flt_trend_actual = (arr_trend_actual[-1] - arr_trend_actual[0]) / run_ 

			# make dictionary
			dict_ = {'feature':col, 'trend_pred':flt_trend_pred, 'trend_act':flt_trend_actual}
			# append to df_empty
			df_empty = df_empty.append(dict_, ignore_index=True)
			# write to csv
			df_empty.to_csv(filename, index=False)

			# create ax
			fig, ax = plt.subplots(figsize=tpl_figsize)
			# plot trendline
			# predicted
			ax.plot(X_train_grouped[col], arr_trend_pred, color='green', label=f'Trend - Predicted ({flt_trend_pred:0.2f})')
			# actual 
			ax.plot(X_train_grouped[col], arr_trend_actual, color='orange', label=f'Trend - Actual ({flt_trend_actual:0.2f})')
			# plot it
			ax.set_title(col)
			# predicted
			ax.plot(X_train_grouped[col], X_train_grouped['predicted'], color='blue', label='Predicted')
			# actual
			ax.plot(X_train_grouped[col], X_train_grouped['actual'], color='red', linestyle=':', label='Actual')
			# legend
			ax.legend(loc='upper right')
			# save fig to self
			self.fig = fig
			# save fig
			plt.savefig(f'{dirname}/{col}.png', bbox_inches='tight')
			# close plot
			plt.close()
		# delete the predicted and actual columns
		del X_train['predicted'], X_train['actual']
		# if logging
		if self.logger:
			self.logger.warning(f'Predicted and actual trends generated and saved to {filename}')



	# define function for combining train and valid
	def combine_train_and_valid(X_train, X_valid, y_train, y_valid):
		# combine train and valid dfs
		self.X_train = pd.concat([X_train, X_valid])
		self.y_train = pd.concat([y_train, y_valid])
		# if using logger
		if self.logger:
			self.logger.warning('Training and validation data combined')
		# print message to console
		print('Training and validation data combined')
		# return
		return self

	# define function to make QQ plot 
	def qq_plot(arr_yhat, ser_actual, filename='./output/plt_qq.png', tpl_figsize=(10,10)):
		# get residuals
		res = arr_yhat - ser_actual
		# make ax
		fig, ax = plt.subplots(figsize=tpl_figsize)
		# title
		ax.set_title('Q-Q Plot')
		# create plot
		sm.qqplot(res, line='45', fit=True, ax=ax)
		# save it
		plt.savefig(filename, bbox_inches='tight')
		# close it
		plt.close()
		# save fig to self
		self.fig = fig
		# log it
		if self.logger:
			self.logger.warning(f'QQ plot saved to {filename}')
		# print message to console
		print(f'QQ plot saved {filename}')
		# return self
		return self

	# define function to get continuous evaluation metrics
	def continuous_eval_metrics(self, model_regressor, X, y):
		# generate predictions
		y_hat = model_regressor.predict(X[model_regressor.feature_names_])
		# explained variance
		exp_var = explained_variance_score(y_true=y, y_pred=y_hat)
		# MAE
		mae = mean_absolute_error(y_true=y, y_pred=y_hat)
		# MSE
		mse = mean_squared_error(y_true=y, y_pred=y_hat)
		# RMSE
		rmse = np.sqrt(mse)
		# put into dictionary
		dict_ = {'exp_var': exp_var,
				   'mae': mae,
				   'mse': mse,
				   'rmse': rmse}
		self.dict_ = dict_
		# if using logger
		if self.logger:
			self.logger.warning('Dictionary of continuous eval metrics generated')
		# print message to console
		print('Dictionary of continuous eval metrics generated')
		# return
		return self

	# define function to get binary eval metrics
	def binary_eval_metrics(self, y, X=None, model_classifier=None, y_hat_class=None, y_hat_proba=None):
		# if starting with model
		if model_classifier:
			# generate predicted class
			y_hat_class = model_classifier.predict(X[model_classifier.feature_names_])
			# generate predicted probabilities
			y_hat_proba = model_classifier.predict_proba(X[model_classifier.feature_names_])[:,1]
		# accuracy
		accuracy = accuracy_score(y_true=y, y_pred=y_hat_class)
		# precision 
		precision = precision_score(y_true=y, y_pred=y_hat_class)
		# recall 
		recall = recall_score(y_true=y, y_score=y_hat_class)
		# f1
		f1 = f1_score(y_true=y, y_pred=y_hat_class)
		# roc auc
		roc_auc = roc_auc_score(y_true=y, y_pred=y_hat_proba)
		# precision recall auc
		pr_auc = average_precision_score(y_true=y, y_pred=y_hat_proba)
		# log loss
		log_loss = log_loss(y_true=y, y_pred=y_hat_proba)
		# brier
		brier = brier_score_loss(y_true=y, y_pred=y_hat_proba)
		# get confusion matrix
		tn, fp, fn, tp = confusion_matrix(y_true=y, y_pred=y_hat_class).ravel() 
		# true positive rate
		tpr = tp / (tp + fn)
		# true negative rate
		tnr = tn / (tn + fp)
		# diagnostic odds ratio
		dor = (tp / fn) / (fp/ tn) 
		# discriminatory power
		disc_pwr = (np.sqrt(3) / np.pi) * (np.log(tpr / (1 - tnr)) + np.log(tnr / (1 - tpr)))
		# gilberts skill score
		# get c
		c = ((tp + fp) * (tp + fn)) / len(y)
		# get gs
		gilb_score = (tp - c) / (tp - c + fn + fp)
		# put into dictionary
		self.dict_ = {'accuracy': accuracy,
				'geometric_mean': geometric_mean,
				'precision': precision,
				'recall': recall,
				'f1': f1,
				'roc_auc': roc_auc,
				'pr_auc': pr_auc,
				'log_loss': log_loss,
				'brier': brier,
				'tpr': tpr,
				'tnr': tnr,
				'dor': dor,
				'disc_pwr': disc_pwr,
				'gilb_score': gilb_score}
		# if logging
		if self.logger:
			logger.warning('Dictionary of binary evaluation metrics generated')
		# print message to console
		print('Dictionary of binary evaluation metrics generated')
		return self




# ========================================================================================
# FUNCTIONS 
# ========================================================================================
# work in progress * 

# get catboost model params
def get_cb_params(model):
	dict_params = model.get_all_params()
	print(model.get_all_params())
	logger.warning(f'CB Params: {dict_params}')
	return dict_params