# model eval
from .model_training_8 import ModelTraining
import sklearn.metrics as skm
import json
import numpy as np
from scipy import stats
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd
from tqdm import tqdm

# define class for model eval
class ModelEval(ModelTraining):
	# initialize
	def __init__(self, cls_model_inference, cls_model_preprocessing, str_dirname_output='./output', str_target='target', str_datecol='date', bool_target_binary=True):
		# initialize parent class
		ModelTraining.__init__(self, cls_model_preprocessing, str_target, str_datecol, str_dirname_output, bool_target_binary)
		# save arguments to object
		self.cls_model_inference = cls_model_inference
		self.cls_model_preprocessing = cls_model_preprocessing
		self.str_dirname_output = str_dirname_output
		self.str_target = str_target
		self.str_datecol = str_datecol
		self.bool_target_binary = bool_target_binary
	# generate predictions
	def generate_predictions(self, X_test):
		# predict
		self.y_hat = self.cls_model_inference.predict(X_test[self.cls_model_inference.feature_names_])
		# if doing classification
		if self.bool_target_binary:
			# probability
			self.y_hat_proba = self.cls_model_inference.predict_proba(X_test[self.cls_model_inference.feature_names_])[:,1]
		# return object
		return self
	# get eval metrics
	def generate_eval_metrics(self, y_test):
		# if doing classification
		if self.bool_target_binary:
			# dictionary of binary eval metrics
			dict_eval_metrics = {
				'accuracy': skm.accuracy_score(y_true=y_test, y_pred=self.y_hat),
				'precision': skm.precision_score(y_true=y_test, y_pred=self.y_hat),
				'recall': skm.recall_score(y_true=y_test, y_pred=self.y_hat),
				'f1': skm.f1_score(y_true=y_test, y_pred=self.y_hat),
				'roc_auc': skm.roc_auc_score(y_true=y_test, y_score=self.y_hat_proba),
				'pr_auc': skm.average_precision_score(y_true=y_test, y_score=self.y_hat_proba),
				'log_loss': skm.log_loss(y_true=y_test, y_pred=self.y_hat_proba),
			}
		# if doing regression
		else:
			# dictionary of continuous eval metrics
			dict_eval_metrics = {
				'explained_variance': skm.explained_variance_score(y_true=y_test, y_pred=y_hat),
				'mae': skm.mean_absolute_error(y_true=y_test, y_pred=y_hat),
				'mse': skm.mean_squared_error(y_true=y_test, y_pred=y_hat),
				'rmse': np.sqrt(skm.mean_squared_error(y_true=y_test, y_pred=y_hat)),
			}
		# write to .json
		json.dump(dict_eval_metrics, open(f'{self.str_dirname_output}/dict_eval_metrics.json', 'w'))
		# save to object
		self.dict_eval_metrics = dict_eval_metrics
		# return object
		return self
	# calculate slope
	def calculate_slope(self, X_test):
		# silence the divide warning
		np.seterr(invalid='ignore')
		# get y_hat
		if self.bool_target_binary:
			y_hat = self.y_hat_proba
		else:
			y_hat = self.y_hat
		# iterate to get progress bar
		list_dict_row = []
		for col in tqdm (X_test.columns):
			# get data type
			str_dtype = X_test[col].dtype
			# if numeric
			if str_dtype == 'float64':
				# calculate slope
				try:
					# replace any possible infs with 0
					ser_x_new = X_test[col].replace([np.inf, -np.inf], 0, inplace=False)
					# get min because cb imputes min
					val_min = np.min(ser_x_new)
					# fillna
					ser_x_new = ser_x_new.fillna(val_min, inplace=False)
					# create df
					df = pd.DataFrame({'feature': ser_x_new, 'y_hat': y_hat})
					# get zscore
					df['zscore'] = np.abs(stats.zscore(df['feature']))
					# subset
					df = df[df['zscore']<=3]
					# calculate slope
					flt_slope = np.polyfit(df['feature'], df['y_hat'], 1)[0]
				except Exception as e:
					# slope
					flt_slope = f'UNKNOWN: {e}'
			# if non-numeric
			else:
				# slope
				flt_slope = np.nan
			# dict row
			dict_row = {'feature': col, 'slope': flt_slope}
			# append
			list_dict_row.append(dict_row)
		# make data frame
		df_trends = pd.DataFrame(list_dict_row)
		# write to csv
		df_trends.to_csv(f'{self.str_dirname_output}/df_trends.csv', index=False)
		# save to object
		self.df_trends = df_trends
		# join to importance df
		df_feat_imp = pd.merge(
			left=self.df_feat_imp,
			right=df_trends,
			left_on='feature',
			right_on='feature',
			how='left',
		)
		# save to .csv
		df_feat_imp.to_csv(f'{self.str_dirname_output}/df_feat_imp.csv', index=False)
		# save to object
		self.df_feat_imp = df_feat_imp
		# return object
		return self
	# precision-recall curve
	def get_precision_recall_curve(self, y_test):
		# get precision rate and recall rate
		precision_r, recall_r, thresholds = skm.precision_recall_curve(y_test, self.y_hat_proba)
		# get the value of chance (i.e., deliquency rate)
		flt_chance = np.mean(y_test)
		# create ax
		fig, ax = plt.subplots(figsize=(10,10))
		# title
		ax.set_title(f"PR Curve: F1 = {self.dict_eval_metrics['f1']:0.4}; AUC = {self.dict_eval_metrics['pr_auc']:0.4}")
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
		# save fig
		plt.savefig(f'{self.str_dirname_output}/plt_pr_curve.png', bbox_inches='tight')
		# show
		plt.show()
		# close
		plt.close()
		# return object
		return self
	# roc-auc curve
	def get_roc_auc_curve(self, y_test):
		# get false positive rate, true positive rate
		fpr, tpr, thresholds = skm.roc_curve(y_true=y_test, y_score=self.y_hat_proba)
		# set up subplots
		fig, ax = plt.subplots(figsize=(10,10))
		# set title
		ax.set_title(f"ROC Plot - (AUC: {self.dict_eval_metrics['roc_auc']:0.4})")
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
		# save fig
		plt.savefig(f'{self.str_dirname_output}/plt_roc_auc.png', bbox_inches='tight')
		# show
		plt.show()
		# close
		plt.close()
		# return object
		return self
	# residual plot
	def get_residual_plot(self, y_test):
		# get residuals
		ser_residuals = self.y_hat - y_test
		# get the norm
		norm = np.linalg.norm(ser_residuals)
		# normalize residuals
		ser_residuals = ser_residuals / norm
		# create ax
		fig, ax = plt.subplots(figsize=(10,10))
		# title
		ax.set_title('Residual Plot (predicted - actual)')
		# distplot
		sns.distplot(ser_residuals, ax=ax)
		# save
		plt.savefig(f'{self.str_dirname_output}/plt_residuals.png', bbox_inches='tight')
		# show
		plt.show()
		# close
		plt.close()
		# return object
		return self
	# qq plot
	def get_qq_plot(self, y_test):
		# get residuals
		ser_residuals = self.y_hat - y_test
		# make ax
		fig, ax = plt.subplots(figsize=(10,10))
		# title
		ax.set_title('Q-Q Plot')
		# create plot
		sm.qqplot(ser_residuals, line='45', fit=True, ax=ax)
		# save it
		plt.savefig(f'{self.str_dirname_output}/plt_qq.png', bbox_inches='tight')
		# show
		plt.show()
		# close it
		plt.close()
		# return object
		return self
	# time to preprocess + predict
	def time_pipeline(self, X_test):
		# start timer
		time_start = time.perf_counter()
		# preprocess
		X_test = self.cls_model_preprocessing.transform(X_test)
		# predict 
		y_hat = self.cls_model_inference.predict(X_test[self.cls_model_inference.feature_names_])
		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# put into dictionary
		self.dict_eval_metrics['flt_sec'] = flt_sec
		# write to .json
		json.dump(self.dict_eval_metrics, open(f'{self.str_dirname_output}/dict_eval_metrics.json', 'w'))
		# return object
		return self