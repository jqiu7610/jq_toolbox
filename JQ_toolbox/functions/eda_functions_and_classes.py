# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 10:33:05 2021

@author: jqiu
"""
import json
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import is_numeric_dtype
import numpy as np
import time
import matplotlib.pyplot as plt
import math
import seaborn as sns


#================================================================================
# CLASSES
#================================================================================

class EDAClass():
	# init
	def __init__(self, logger):
		self.logger = logger
	
	"""
	define function that gets shape
	"""
	def get_shape(self, df):
		shape = df.shape
		self.shape = shape
		# log it
		if self.logger:
			logger.warning(f'Data shape is {shape}')
		print(f'Data shape is {shape}')
		return self
	
	"""
	define function to remove columns with no variance		
	only need to call list_no_var, df will get changed automatically once you call for list
	"""
	def drop_no_variance(self, df):
		# create empty list
		list_no_var = []
		# iterate through columns in df
		for col in df.columns:
			# get the series
			series_ = df[col]
			# drop na
			series_.dropna(inplace=True)
			# get count unique
			count_unique = series_.nunique()
			if count_unique ==1:
				# append to list
				list_no_var.append(col)
			self.list_no_var = list_no_var
		# drop list_no_var
		df = df.drop(list_no_var, axis=1, inplace=True)
		# if logger
		if self.logger:
		# log it
			self.logger.warning('list of no-variance columns generated and removed from dataframe')
		print('list of no-variance columns generated and removed from dataframe')
		# return
		return self

	"""
	define function to generate a missing proportions df
	"""
	def get_pct_missing(self, df, str_filename='./output/pct_missing.csv'):
		# find percent missing for columns
		percent_missing = df.isnull().sum() * 100 / len(df)
		missing_value_df = pd.DataFrame({'column_name': df.columns, 'percent_missing':percent_missing})
		self.missing_value_df = missing_value_df
		# log it if logger
		if self.logger:
			self.logger.warning('missing proportions found')
		print('missing proportions found')
		# write to csv
		missing_value_df.to_csv(str_filename, index=False)
		return self

	"""
	define funciton to drop columns from user-fed list 
	"""
	def drop_cols(self, df, list_drop_cols):
		# future proof
		list_drop_cols = [col for col in list_drop_cols if col in list(df.columns)]
		self.df = df.drop(list_drop_cols, axis=1)
		count = len(list_drop_cols)
		#log
		if self.logger:
			self.logger.warning(f'{count} columns have been removed from the dataframe')
		print(f'{count} columns have been removed from the dataframe')
		#return df
		return self

	"""
	list non-numeric columns
	"""
	def get_list_nonnumeric(self, df):
		# define non numeric
		self.list_non_numeric = [col for col in df.columns if is_numeric_dtype(df[col])==False]
		# log it
		if self.logger:
			self.logger.warning('non-numeric generated')
		print('list non-numeric generated')
		return self

	"""
	define function to list numeric
	"""
	def get_list_numeric(self, df):
		# define numeric
		self.list_numeric = [col for col in df.columns if is_numeric_dtype(df[col])==True]
		# log it
		if self.logger:
			self.logger.warning('list numeric generated')
		print('list numeric generated')
		return self

	"""
	define function that logs df info
	X = EDAPlotting(logger=logger)
	X.log_df_info(df=df, str_dflogname='fakelog2', str_datecol='col1', str_bin_target='col5', bool_low_memory=False)
	"""
	def log_df_info(df, str_dflogname='df_train', str_datecol='dtmStampCreation__app', str_bin_target='TARGET__app',
				 bool_low_memory=False):
		# get rows
		int_nrows = df.shape[0]
		# get cols
		int_ncols = df.shape[1]
		# logic
		if bool_low_memory: 
			int_n_missing_all = 0 
			# iterate through cols
			for a, col in enumerate(df.columns):
				# print message
				print(f'Checking NaN: {a+1}/{int_ncols}')
				# get number missing per column
				int_n_missing_col = df[col].isnull().sum()
				# add to int_n_missing_all
				int_n_missing_all += int_n_missing_col 
			# get roportion NaN
			flt_prop_nan = int_n_missing_all/(int_nrows*int_ncols)
		else:
			# get proportion NaN
			flt_prop_nan = np.sum(df.isnull().sum())/(int_nrows*int_ncols)
		# get minimum str_datecol
		min_ = np.min(df[str_datecol])
		# get max str_datecol
		max_ = np.max(df[str_datecol])
		# get delinquency rate
		flt_pro_delinquent = np.mean(df[str_bin_target])
		# if logger
		if self.logger:
			self.logger.warning(f'{str_dflogname}: {int_nrows} rows, {int_ncols} columns')
			self.logger.warning(f'{str_dflogname}: {flt_prop_nan:0.3f} NaN')
			self.logger.warning(f'{str_dflogname}: Min {str_datecol} = {min_}')
			self.logger.warning(f'{str_dflogname}: Max {str_datecol} = {max_}')
			self.logger.warning(f'{str_dflogname}: Target Proportion = {flt_pro_delinquent:0.3f}')
		print(f'{str_dflogname}: {int_nrows} rows, {int_ncols} columns')
		print(f'{str_dflogname}: {flt_prop_nan:0.3f} NaN')
		print(f'{str_dflogname}: Min {str_datecol} = {min_}')
		print(f'{str_dflogname}: Max {str_datecol} = {max_}')
		print(f'{str_dflogname}: Target Proportion = {flt_pro_delinquent:0.3f}')	

"""
class that takes care of most EDA plotting needs
"""
class EDAPlotting():
	# init
	def __init__(self, logger):
		self.logger = logger
	
	"""
	define function to show pie plot of missing vs non-missing ***save as png***
	"""
	def plot_na_overall(self, df, str_filename, tpl_figsize=(10,15)):
		"""
		takes df and returns pie chart proportion of missing
		"""
		# get total number of missing
		n_missing = np.sum(df.isnull().sum())
		# get total observations
		n_observations = df.shape[0] * df.shape[1]
		# both into a list
		list_values = [n_missing, n_observations]
		# create axis
		fig, ax = plt.subplots(figsize=tpl_figsize)
		# title
		ax.set_title('Pie chart of Missing Proportions')
		ax.pie(x=[n_missing, n_observations],
				colors=['r', 'c'],
				explode=(0, 0.1),
				labels=['Missing', 'Non-Missing'],
				autopct='%1.1f%%')
		# save fig to self
		self.fig = fig
		# save fig 
		plt.savefig(str_filename, bbox_inches='tight')
		# show plot
		plt.show()
		# close plot
		plt.close()
		if self.logger:
			self.logger.warning('pie chart for missing proportions generated')
		# return
		return self
	
	"""
	define function to compare binary inputs 0 vs 1 ***save as png***
	"""
	def plot_binary_comparison(self, df_col, str_filename='./output/target_comparison.png'):
		# get value counts for each
		ser_val_counts = pd.value_counts(df_col)
		# get x
		x = ser_val_counts.index
		# get y
		y = ser_val_counts.values
		# get total 
		int_total = len(df_col)
		# get pct for zero
		flt_pct_missed = (y[1]/int_total)*100
		# get pct for one
		flt_pct_made = (y[0]/int_total)*100
		# create axis
		fig, ax = plt.subplots(figsize=(15,10))
		# title
		ax.set_title(f'{flt_pct_made:0.3f}% = bitZERO, {flt_pct_missed:0.3f}% = bitONE, (N = {int_total})')
		# frequency bar plot
		ax.bar(x, y)
		# ylabel
		ax.set_ylabel('Frequency')
		# xticks
		ax.set_xticks([0,1])
		# xtick labels
		ax.set_xticklabels(['0', '1'])
		# save
		plt.savefig(str_filename, bbox_inches='tight')
		# show plot
		plt.show()
		# save fig to self
		self.fig = fig
		# log it
		if self.logger:
			self.logger.warning(f'target frequency plot saved to {str_filename}')
		print(f'target frequency plot saved to {str_filename}')
		# return
		return self
	
	"""
	define function to create dtype frequency plot
	"""
	def plot_dtype(self, df, str_filename='./output/plt_dtype.png', tpl_figsize=(10,10)):
		# get frequency of numeric and non-numeric
		counter_numeric = 0
		counter_non_numeric = 0
		for col in df.columns:
			if is_numeric_dtype(df[col]):
				counter_numeric += 1
			else:
				counter_non_numeric += 1 
		# number of columns
		int_n_cols = len(list(df.columns))
		# proportion numeric
		flt_pct_numeric = (counter_numeric / int_n_cols) * 100
		# proportion non-numeric
		flt_pct_nonnumeric = (counter_non_numeric / int_n_cols) * 100
		# create ax
		fig, ax = plt.subplots(figsize=tpl_figsize)
		# create title
		ax.set_title(f'{flt_pct_numeric:0.3f}% Numeric, {flt_pct_nonnumeric:0.3f}% Non-Numeric, (N = {int_n_cols})')
		# set y label
		ax.set_ylabel('Frequency')
		# bar plot
		ax.bar(['Numeric', 'Non-Numeric'], [counter_numeric, counter_non_numeric])
		# save plot
		plt.savefig(str_filename, bbox_inches='tight')
		# show plot
		plt.show()
		# close plot
		plt.close()
		# set fig to self
		self.fig = fig
		# log
		if self.logger:
			self.logger.warning(f'Data type frequency plot saved to {str_filename}')
		print(f'Data type frequency plot saved to {str_filename}')
		# return
		return self 

"""
define DropNoVariance
X = DropNoVariance(list_cols=list_cols)
X.fit(X=df)
X.transform(X=df)
"""
class DropNoVariance(BaseEstimator, TransformerMixin):
	# initialize class
	def __init__(self, list_cols, bool_low_memory=True):
		self.list_cols = list_cols
		self.bool_low_memory = bool_low_memory
	# fit to X
	def fit(self, X, y=None):
		# if we have low memory
		if self.bool_low_memory:
			# instantiate empty list
			list_novar = []
			# iterate through cols
			for a, col in enumerate(self.list_cols):
				# print message
				print(f'Checking col {a+1}/{len(self.list_cols)}')
				# get number of unique
				n_unique = len(pd.value_counts(X[col]))
				# logic to identify no variance cols
				if n_unique == 1:
					list_novar.append(col)
		else:
			# define helper function
			def GET_NUNIQUE(ser_):
				n_unique = len(pd.value_counts(ser_))
				return n_unique
			# apply function to every column
			ser_nunique = X[self.list_cols].apply(lambda x: GET_NUNIQUE(ser_=x), axis=0)
			# get the cols with nunique == 1
			list_novar = list(ser_nunique[ser_nunique==1].index)
		# save to object
		self.list_novar = list_novar
		# return self
		return self
	# transform X
	def transform(self, X):
		# make sure all cols in self.list_novar are in X
		list_cols = [col for col in self.list_novar if col in list(X.columns)]
		# drop list_cols
		X.drop(list_cols, axis=1, inplace=True)
		# return
		return X

"""
define class for redundant feature dropper 
X = DropRedundantFeatures(list_cols=list_cols,int_n_rows_check=100)
X.fit(X=df)
X.transform(X=df) will get results
"""
class DropRedundantFeatures(BaseEstimator, TransformerMixin):
	# initialize class
	def __init__(self, list_cols, int_n_rows_check=10000):
		self.list_cols = list_cols
		self.int_n_rows_check = int_n_rows_check
	# fit
	def fit(self, X, y=None):
		# instantiate empty list
		list_redundant_cols = []
		for a, cola in enumerate(self.list_cols):
			# status message
			print(f'Currently, there are {len(list_redundant_cols)} redundant columns.')
			# status message
			print(f'Checking column {a+1}/{len(self.list_cols)}')
			# logic
			if cola not in list_redundant_cols:
				# iterate through the other cols
				for colb in self.list_cols[a+1:]:
					# check if subset of cola == colb
					if X[cola].iloc[:self.int_n_rows_check].equals(X[colb].iloc[:self.int_n_rows_check]):
						# print message
						print(f'First {self.int_n_rows_check} rows in {colb} are redundant with {cola}')
						# check if the whole column is redundant
						if X[cola].equals(X[colb]):
							# print message
							print(f'After checking all rows, {colb} is redundant with {cola}')
							list_redundant_cols.append(colb)
						else:
							print(f'After checking all rows, {colb} is not redundant with {cola}')
		# save to object
		self.list_redundant_cols = list_redundant_cols
		# return
		return self
	# transform
	def transform(self, X):
		# make sure all cols in self.list_redundant_cols are in X
		list_cols = [col for col in self.list_redundant_cols if col in list(X.columns)]
		# drop list_cols
		X.drop(list_cols, axis=1, inplace=True)
		# return
		return X


"""
define class for automating distribution plot analysis
"""
class DistributionAnalysis(BaseEstimator, TransformerMixin):
	# initialiaze
	def __init__(self, list_cols, int_nrows=10000, int_random_state=42, flt_thresh_upper=0.95, tpl_figsize=(10,10), 
		         str_dirname='./output/distplots'):
		self.list_cols = list_cols
		self.int_nrows = int_nrows
		self.int_random_state = int_random_state
		self.flt_thresh_upper = flt_thresh_upper
		self.tpl_figsize = tpl_figsize
		self.str_dirname = str_dirname
	# random sample
	def get_random_sample(self, X, str_df_name='train'):
		# logic
		if str_df_name == 'train':
			self.df_train_sub = X.sample(n=self.int_nrows, random_state=self.int_random_state)
		elif str_df_name == 'valid':
			self.df_valid_sub = X.sample(n=self.int_nrows, random_state=self.int_random_state)
		else:
			self.df_test_sub = X.sample(n=self.int_nrows, random_state=self.int_random_state)
	# compare each col
	def fit(self, X, y=None):
		# iterate through cols
		list_sig_diff = []
		for a, col in enumerate(self.list_cols):
			# print
			print(f'Currently {len(list_sig_diff)} columns with a significant difference')
			# print
			print(f'Evaluating col {a+1}/{len(self.list_cols)}')
			# create a df with just the cols
			df_col = pd.DataFrame({'train': list(self.df_train_sub[col]),
				                   'valid': list(self.df_valid_sub[col]),
				                   'test': list(self.df_test_sub[col])})
			# get number of rows per sample
			int_len_sample = int(self.int_nrows/100) # always doing 100 samples
			# create list to use for sample
			list_empty = []
			for b in range(100): # always doing 100 samples
				# create list containing value for b the same length as a samplel
				list_ = list(itertools.repeat(b, int_len_sample))
				# extend list_empty
				list_empty.extend(list_)
			# create a dictionary to use for grouping
			dict_ = dict(zip(list(df_col.columns), ['median' for col in list(df_col.columns)]))
			# make list_empty into a column in df_col
			df_col['sample'] = list_empty
			# group df_col by sample and get median for each of 100 samples
			df_col = df_col.groupby('sample', as_index=False).agg(dict_)
			# TRAIN VS. VALID
			# first test (train > valid)
			flt_avg = np.mean(df_col.apply(lambda x: 1 if x['train'] > x['valid'] else 0, axis=1))
			# logic for significance
			if (flt_avg >= self.flt_thresh_upper):
				# print
				print(f'Significant difference in {col} between train and valid ({flt_avg:0.4})')
				# append to list
				list_sig_diff.append(col)
				# make distribution plot
				fig, ax = plt.subplots(figsize=self.tpl_figsize)
				# title
				ax.set_title(f'{col} - Train > Test')
				# plot train
				sns.distplot(df_col['train'], kde=True, color="r", ax=ax, label='Train')
				# plot valid
				sns.distplot(df_col['valid'], kde=True, color="g", ax=ax, label='Valid')
				# plot test
				sns.distplot(df_col['test'], kde=True, color="b", ax=ax, label='Test')
				# legend
				plt.legend()
				# save plot
				plt.savefig(f'{self.str_dirname}/{col}.png', bbox_inches='tight')
				# close plot
				plt.close()
				# move to next col
				continue
			else:
				# second test (valid > train)
				flt_avg = np.mean(df_col.apply(lambda x: 1 if x['valid'] > x['train'] else 0, axis=1))
				# logic for significance
				if (flt_avg >= self.flt_thresh_upper):
					# print
					print(f'Significant difference in {col} between train and valid ({flt_avg:0.4})')
					# append to list
					list_sig_diff.append(col)
					# make distribution plot
					fig, ax = plt.subplots(figsize=self.tpl_figsize)
					# title
					ax.set_title(f'{col} Valid > Train')
					# plot train
					sns.distplot(df_col['train'], kde=True, color="r", ax=ax, label='Train')
					# plot valid
					sns.distplot(df_col['valid'], kde=True, color="g", ax=ax, label='Valid')
					# plot test
					sns.distplot(df_col['test'], kde=True, color="b", ax=ax, label='Test')
					# legend
					plt.legend()
					# save plot
					plt.savefig(f'{self.str_dirname}/{col}.png', bbox_inches='tight')
					# close plot
					plt.close()
					# move to next col
					continue
			# TRAIN VS. TEST
			# first test (train > test)
			flt_avg = np.mean(df_col.apply(lambda x: 1 if x['train'] > x['test'] else 0, axis=1))
			# logic for significance
			if (flt_avg >= self.flt_thresh_upper):
				# print
				print(f'Significant difference in {col} between train and test ({flt_avg:0.4})')
				# append to list
				list_sig_diff.append(col)
				# make distribution plot
				fig, ax = plt.subplots(figsize=self.tpl_figsize)
				# title
				ax.set_title(f'{col} - Train > Test')
				# plot train
				sns.distplot(df_col['train'], kde=True, color="r", ax=ax, label='Train')
				# plot valid
				sns.distplot(df_col['valid'], kde=True, color="g", ax=ax, label='Valid')
				# plot test
				sns.distplot(df_col['test'], kde=True, color="b", ax=ax, label='Test')
				# legend
				plt.legend()
				# save plot
				plt.savefig(f'{self.str_dirname}/{col}.png', bbox_inches='tight')
				# close plot
				plt.close()
				# move to next col
				continue
			else:
				# second test (test > train)
				flt_avg = np.mean(df_col.apply(lambda x: 1 if x['test'] > x['train'] else 0, axis=1))
				# logic for significance
				if (flt_avg >= self.flt_thresh_upper):
					# print
					print(f'Significant difference in {col} between train and test ({flt_avg:0.4})')
					# append to list
					list_sig_diff.append(col)
					# make distribution plot
					fig, ax = plt.subplots(figsize=self.tpl_figsize)
					# title
					ax.set_title(f'{col} - Test > Train')
					# plot train
					sns.distplot(df_col['train'], kde=True, color="r", ax=ax, label='Train')
					# plot valid
					sns.distplot(df_col['valid'], kde=True, color="g", ax=ax, label='Valid')
					# plot test
					sns.distplot(df_col['test'], kde=True, color="b", ax=ax, label='Test')
					# legend
					plt.legend()
					# save plot
					plt.savefig(f'{self.str_dirname}/{col}.png', bbox_inches='tight')
					# close plot
					plt.close()
					# move to next col
					continue
			# VALID VS. TEST
			# first test (valid > test)
			flt_avg = np.mean(df_col.apply(lambda x: 1 if x['valid'] > x['test'] else 0, axis=1))
			# logic for significance
			if (flt_avg >= self.flt_thresh_upper):
				# print
				print(f'Significant difference in {col} between valid and test ({flt_avg:0.4})')
				# append to list
				list_sig_diff.append(col)
				# make distribution plot
				fig, ax = plt.subplots(figsize=self.tpl_figsize)
				# title
				ax.set_title(f'{col} - Valid > Test')
				# plot train
				sns.distplot(df_col['train'], kde=True, color="r", ax=ax, label='Train')
				# plot valid
				sns.distplot(df_col['valid'], kde=True, color="g", ax=ax, label='Valid')
				# plot test
				sns.distplot(df_col['test'], kde=True, color="b", ax=ax, label='Test')
				# legend
				plt.legend()
				# save plot
				plt.savefig(f'{self.str_dirname}/{col}.png', bbox_inches='tight')
				# close plot
				plt.close()
			else:
				# second test (test > valid)
				flt_avg = np.mean(df_col.apply(lambda x: 1 if x['test'] > x['valid'] else 0, axis=1))
				# logic for significance
				if (flt_avg >= self.flt_thresh_upper):
					# print
					print(f'Significant difference in {col} between test and valid ({flt_avg:0.4})')
					# append to list
					list_sig_diff.append(col)
					# make distribution plot
					fig, ax = plt.subplots(figsize=self.tpl_figsize)
					# title
					ax.set_title(f'{col} - Test > Valid')
					# plot train
					sns.distplot(df_col['train'], kde=True, color="r", ax=ax, label='Train')
					# plot valid
					sns.distplot(df_col['valid'], kde=True, color="g", ax=ax, label='Valid')
					# plot test
					sns.distplot(df_col['test'], kde=True, color="b", ax=ax, label='Test')
					# legend
					plt.legend()
					# save plot
					plt.savefig(f'{self.str_dirname}/{col}.png', bbox_inches='tight')
					# close plot
					plt.close()	
		# save to object
		self.list_sig_diff = list_sig_diff
		# delete the objects we don't want
		del self.df_train_sub, self.df_valid_sub, self.df_test_sub, df_col
		# return self
		return self
	# drop columns
	def transform(self, X):
		# make sure all cols in self.list_sig_diff are in X
		list_cols = [col for col in self.list_sig_diff if col in list(X.columns)]
		# drop list_cols
		X.drop(list_cols, axis=1, inplace=True)
		# return
		return X

"""
class for plot comparisons for continuous targets
"""
class Continuous_Target_Comparison:
	# get y
	def get_y(self, ser_y, str_df_name='train'):
		if str_df_name == 'train':
			self.y_train = ser_y
		elif str_df_name == 'valid':
			self.y_valid = ser_y
		elif str_df_name == 'test':
			self.y_test = ser_y
		else:
			raise Exception('str_df_name must be "train", "valid", or "test"')

	# make plot
	def create_plot(self, str_filename='./output/plt_target_comparison.png', tpl_figsize=(12,8)):
		# make ax
		fig, ax = plt.subplots(figsize=tpl_figsize)
		# title
		ax.set_title('Continuous Target Distributions')
		# train
		sns.distplot(self.y_train, ax=ax, kde=True, label='Train', color='r')
		# valid
		sns.distplot(self.y_valid, ax=ax, kde=True, label='Valid', color='g')
		# test
		sns.distplot(self.y_test, ax=ax, kde=True, label='Test', color='b')
		# legend
		plt.legend()
		# save
		plt.savefig(str_filename, bbox_inches='tight')
		# show plot
		plt.show()
		# close plot
		plt.close()

"""
create class that checks for common columns
"""
class Common_Column_Checker:
	# get cols
	def get_cols(self, list_cols, str_df_name):
		# logic for saving lists
		if str_df_name == 'train':
			self.list_train = list_cols[:]
		elif str_df_name == 'valid':
			self.list_valid = list_cols[:]
		elif str_df_name == 'test':
			self.list_test = list_cols[:]
		else:
			raise Exception('str_df_name must be "train", "valid", or "test"')
		# return
		return self

	# create table
	def create_table(self):
		# train and train
		int_train_train = len(self.list_train)
		# train and valid
		int_train_valid = len([col for col in self.list_train if col in self.list_valid])
		# train and test
		int_train_test = len([col for col in self.list_train if col in self.list_test])

		# valid and train
		int_valid_train = len([col for col in self.list_valid if col in self.list_train])
		# valid and valid
		int_valid_valid = len(self.list_valid)
		# valid and test
		int_valid_test = len([col for col in self.list_valid if col in self.list_test])

		# test and train
		int_test_train = len([col for col in self.list_test if col in self.list_train])
		# test and valid
		int_test_valid = len([col for col in self.list_test if col in self.list_valid])
		# int_valid_valid
		int_test_test = len(self.list_test)

		# make data frame
		self.df_table = pd.DataFrame({'df':['train', 'valid', 'test'],
										'train':[int_train_train, int_train_valid, int_train_test],
										'valid':[int_valid_train, int_valid_valid, int_valid_test],
										'test':[int_test_train, int_test_valid, int_test_test]})
		# return
		return self
	# save table
	def save_table(self, str_filename='./output/df_common_cols.csv'):
		# write to csv
		self.df_table.to_csv(str_filename, index=False)

#===============================================================================
# FUNCTIONS
#===============================================================================
# define function for logging
def LOG_EVENTS(str_filename='./logs/eda_pt1.log'):
	# set logging format
	FORMAT = '%(name)s:%(levelname)s:%(asctime)s:%(message)s'
	# get logger
	logger = logging.getLogger(__name__)
	# try making log
	try:
		# reset any other logs
		handler = logging.FileHandler(str_filename, mode='w')
	except FileNotFoundError:
		os.mkdir('./logs')
		# reset any other logs
		handler = logging.FileHandler(str_filename, mode='w')
	# change to append
	handler = logging.FileHandler(str_filename, mode='a')
	# set the level to info
	handler.setLevel(logging.INFO)
	# set format
	formatter = logging.Formatter(FORMAT)
	# format the handler
	handler.setFormatter(formatter)
	# add handler
	logger.addHandler(handler)
	# return logger
	return logger

# define function to check out shape of df
def GET_SHAPE(df, logger=None):
	shape = df.shape
	#log it
	if logger:
		logger.warning(f'Data shape is {shape}')

# define function to read chunks of CSV
def CHUNKY_CSV(str_filename, chunksize, logger=None):
	# start timer
	time_start = time.perf_counter()
	# chunk csv
	df = pd.DataFrame()
	for chunk in pd.read_csv(str_filename, chunksize=chunksize, low_memory=False):
		df = pd.concat([df,chunk])
	# log it
	if logger:
		logger.warning(f'Data imported from csv chunks in {(time.perf_counter()-time_start)/60:0.3f} min.')
	return df


#df = pd.DataFrame(np.array([[1,2,4,4],[1,1,4,4],[6,7,4,4],[1,1,4,4],[1,1,4,4]]), columns=['a','b','c','c'])
# define function to read json
def JSON_TO_DF(str_filename, logger=None):
	# start timer
	time_start = time.perf_counter()
	# read json file
	df = json.load(open(str_filename, 'r'))
	# store in df
	df = pd.DataFrame.from_dict(df, orient='columns')
	# if we are using a logger
	if logger:
		# log it
		logger.warning(f'Data imported from json in {(time.perf_counter()-time_start)/60:0.3f} min.')
	# return
	return df

# define function to read csv
def READ_CSV(str_filename, logger=None):
	#start timer
	time_start = time.perf_counter()
	#read csv
	df = pd.read_csv(str_filename)
	#log it
	if logger:
		logger.warning(f'Data imported from csv in {(time.perf_counter()-time_start)/60:0.3f} min.')
	# return
	return df

# define function to write csv
def WRITE_CSV(df, str_filename='./output/df_raw_eda.csv', logger=None):
	#start timer
	time_start = time.perf_counter()
	df.to_csv(str_filename, index=False)
	# log it
	if logger:
		logger.warning(f'Data written to csv in {(time.perf_counter()-time_start)/60:0.3f} min.')
	
	
# define function to remove columns with no variance 
def DROP_NO_VARIANCE(df, logger=None):
	# create empty list
	list_no_var = []
	# iterate through columns in df
	for col in df.columns:
		# get the series
		series_ = df[col]
		# drop na
		series_.dropna(inplace=True)
		# get count unique
		count_unique = series_.nunique()
		# if count_unique == 1
		if count_unique == 1:
			# append to list
			list_no_var.append(col)
	# drop list_no_var
	df = df.drop(list_no_var, axis=1, inplace=True)
	# log it
	if logger:
		logger.warning(f'list of no-variance columns generated and removed from dataframe')
	return list_no_var, df

# define function to plot % of missing from entire df
def PIE_PLOT_NA_OVERALL(df, str_filename, tpl_figsize=(10,15), logger=None):
	"""
	takes a data frame and returns a pie chart of missing and not missing.
	"""
	# get total number missing
	n_missing = np.sum(df.isnull().sum())
	# get total observations
	n_observations = df.shape[0] * df.shape[1]
	# both into a list
	list_values = [n_missing, n_observations]
	# create axis
	fig, ax = plt.subplots(figsize=tpl_figsize)
	# title
	ax.set_title('Pie Chart of Missing Values')
	ax.pie(x=[n_missing, n_observations], 
	       colors=['r', 'c'],
	       explode=(0, 0.1),
	       labels=['Missing', 'Non-Missing'], 
	       autopct='%1.1f%%')
	# save fig
	plt.savefig(str_filename, bbox_inches='tight')
	# close plot
	plt.close()
	if logger:
		logger.warning('pie chart for missing % generated')
	# return fig
	return fig

# define function to generate missing proportions df
def GET_PCT_MISSING(df, str_filename, logger=None):
	# find percent missing for columns
	percent_missing = df.isnull().sum() * 100 / len(df)
	missing_value_df = pd.DataFrame({'column_name': df.columns,
								  'percent_missing': percent_missing})
	if logger:
		logger.warning('missing proportions found')
	# write to csv
	missing_value_df.to_csv(str_filename, index=False)
	return missing_value_df

# manually search and remove func, ***EXACT MATCH, OR KEY WORD FOR COLUMN NAME***
def SEARCH_N_DESTROY(df, str_name, logger=None):
	# get desired columns into a list
	list_col = []
	# for loop for cols in df
	for col in df:
	    if str_name in col:
	        list_col.append(col)
	# count the list
	count = len(list_col)
	# drop
	df.drop(list_col, axis=1, inplace=True)
	# log
	if logger:
	    logger.warning(f"{count} columns removed with column name key-word '{str_name}'")
	# return
	return df

# define frunciton to drop duplicated and unnecessary columns
def DROP_COLS(df, list_drop_cols, logger=None):
	df = df.drop(list_drop_cols, axis=1)
	count = len(list_drop_cols)
	#log
	if logger:
		logger.warning(f'{count} columns have been removed from the dataframe')
	#return df
	return df



# define function to get rid of all duplicate rows
def RID_DUPES(df, logger=None):
	# get list of dupe rows, keep first unique as false
	list_dup_rows = df.duplicated(keep='first')
	# count the number of dups
	count_dup_rows = sum(list_dup_rows)
	# drop the dup rows
	df = df.drop_duplicates()
	# log it
	logger.warning(f'{count_dup_rows} duplicated rows eliminated')
	# get list dupe cols, keep first unique as false
	list_dup_cols = df.columns.duplicated(keep='first')
	# count number of dup cols
	count_dup_cols = sum(list_dup_cols)
	# drop dup cols
	df = df.loc[:,~df.columns.duplicated()]
	# log it
	if logger:
		logger.warning(f'{count_dup_cols} duplicated cols eliminated')
	# return 
	return df


# list numeric and list non-numeric columns
def GET_NUMERIC(df, logger=None):
	# define non numeric
	list_non_numeric = [col for col in df.columns if is_numeric_dtype(df[col])==False]
	# define numeric
	list_numeric = [col for col in df.columns if is_numeric_dtype(df[col])==True]  
	# log it
	if logger:
		logger.warning('list numeric and list non-numeric generated')
	return list_non_numeric, list_numeric



# define function to compare made/missed payments
def PLOT_BINARY_COMPARISON(ser_bin, str_filename='./output/target_freqplot.png', logger=None):
	# get value counts for each
	ser_val_counts = pd.value_counts(ser_bin)
	# get x
	x = ser_val_counts.index
	# get y
	y = ser_val_counts.values
	# get total
	int_total = len(ser_bin)
	# get pct missed
	flt_pct_missed = (y[1]/int_total)*100
	# get proportion made
	flt_pct_made = (y[0]/int_total)*100
	# create axis
	fig, ax = plt.subplots(figsize=(15, 10))
	# title
	ax.set_title(f'{flt_pct_made:0.3f}% = 0, {flt_pct_missed:0.3f}% = 1, (N = {int_total})')
	# frequency bar plot
	ax.bar(x, y)
	# ylabel
	ax.set_ylabel('Frequency')
	# xticks
	ax.set_xticks([0, 1])
	# xtick labels
	ax.set_xticklabels(['0','1'])
	# save
	plt.savefig(str_filename, bbox_inches='tight')
	# log it
	if logger: 
		logger.warning(f'target frequency plot saved to {str_filename}')
	# return
	return fig


# define function to log df info
def LOG_DF_INFO(df, str_dflogname='df_train', str_datecol='dtmStampCreation__app', str_bin_target='TARGET__app', 
	            logger=None, bool_low_memory=True):
	# get rows
	int_nrows = df.shape[0]
	# get columns
	int_ncols = df.shape[1]
	# logic
	if bool_low_memory:
		int_n_missing_all = 0
		# iterate through cols
		for a, col in enumerate(df.columns):
			# print message
			print(f'Checking NaN: {a+1}/{int_ncols}')
			# get number missing per col
			int_n_missing_col = df[col].isnull().sum()
			# add to int_n_missing_all
			int_n_missing_all += int_n_missing_col
		# get proportion NaN
		flt_prop_nan = int_n_missing_all/(int_nrows*int_ncols)
	else:
		# get proportion NaN
		flt_prop_nan = np.sum(df.isnull().sum())/(int_nrows*int_ncols)
	# get min str_datecol
	min_ = np.min(df[str_datecol])
	# get max dtmstampCreation__app
	max_ = np.max(df[str_datecol])
	# get deliquency rate
	flt_prop_delinquent = np.mean(df[str_bin_target])
	# if logging
	if logger:
		logger.warning(f'{str_dflogname}: {int_nrows} rows, {int_ncols} columns')
		logger.warning(f'{str_dflogname}: {flt_prop_nan:0.3f} NaN')
		logger.warning(f'{str_dflogname}: Min {str_datecol} = {min_}')
		logger.warning(f'{str_dflogname}: Max {str_datecol} = {max_}')
		logger.warning(f'{str_dflogname}: Target Proportion = {flt_prop_delinquent:0.3f}')


# define function to plot numeric v non-numeric dtype freq
def PLOT_DTYPE(df, str_filename='./output/plt_dtype.png', tpl_figsize=(10,10), logger=None):
	# get frequency of numeric and non-numeric
	counter_numeric = 0
	counter_non_numeric = 0
	for col in df.columns:
		if is_numeric_dtype(df[col]):
			counter_numeric += 1
		else:
			counter_non_numeric += 1
	# number of cols
	int_n_cols = len(list(df.columns))
	# % numeric
	flt_pct_numeric = (counter_numeric / int_n_cols) * 100
	# % non-numeric
	flt_pct_non_numeric = (counter_non_numeric / int_n_cols) * 100
	# create ax
	fig, ax = plt.subplots(figsize=tpl_figsize)
	# title
	ax.set_title(f'{flt_pct_numeric:0.3f}% Numeric, {flt_pct_non_numeric:0.3f}% Non-Numeric (N = {int_n_cols})')
	# y label
	ax.set_ylabel('Frequency')
	# bar plot
	ax.bar(['Numeric','Non-Numeric'], [counter_numeric, counter_non_numeric])
	# save plot
	plt.savefig(str_filename, bbox_inches='tight')
	# show plot
	plt.show()
	# close plot
	plt.close()
	# log
	if logger:
		logger.warning(f'Data type frequency plot saved to {str_filename}')
	# return
	return fig

# define function to save proportion NaN by column
def SAVE_NAN_BY_COL(df, str_filename='./output/df_propna.csv', logger=None, bool_low_memory=True):
	# logic
	if bool_low_memory:
		# empty list
		list_empty = []
		# iterate through cols
		for a, col in enumerate(df.columns):
			# print message
			print(f'Checking NaN: {a+1}/{df.shape[1]}')
			# get prop missing
			flt_prop_nan = df[col].isnull().sum()/len(df[col])
			# create dict
			dict_ = {'column': col,
			         'prop_nan': flt_prop_nan}
			# append to list_empty
			list_empty.append(dict_)
		# make df
		df = pd.DataFrame(list_empty)
	else:
		# get proportion missing by col
		ser_propna = df.isnull().sum()/df.shape[0]
		# put into df
		df = pd.DataFrame({'column': ser_propna.index,
	                       'prop_nan': ser_propna})
	# sort
	df.sort_values(by='prop_nan', ascending=False, inplace=True)
	# save to csv
	df.to_csv(str_filename, index=False)
	# if using logger
	if logger:
		logger.warning(f'csv file of proportion NaN by column generated and saved to {str_filename}')


# define function to get training only
def CHRON_GET_TRAIN(df, flt_prop_train=0.5, logger=None):
	# get n_rows in df
	n_rows_df = df.shape[0]
	# get last row in df_train
	n_row_end_train = math.floor(n_rows_df * flt_prop_train)
	# get training data
	df = df.iloc[:n_row_end_train, :]
	# if using logger
	if logger:
		# log it
		logger.warning(f'Subset df to first {flt_prop_train} rows for training')
	# return
	return df


# define function for chronological split
def CHRON_TRAIN_VALID_TEST_SPLIT(df, flt_prop_train=0.5, flt_prop_valid=0.25, logger=None):
	# get n_rows in df
	n_rows_df = df.shape[0]
	# get last row in df_train
	n_row_end_train = math.floor(n_rows_df * flt_prop_train)
	# get last row in df_valid
	n_row_end_valid = math.floor(n_rows_df * (flt_prop_train + flt_prop_valid))
	# create train, valid, test
	df_train = df.iloc[:n_row_end_train, :]
	df_valid = df.iloc[n_row_end_train:n_row_end_valid, :]
	df_test = df.iloc[n_row_end_valid:, :]
	# calculate proportion in test
	flt_prop_test = 1 - (flt_prop_train + flt_prop_valid)
	# if using logger
	if logger:
		# log it
		logger.warning(f'Split df into train ({flt_prop_train}), valid ({flt_prop_valid}), and test ({flt_prop_test})')
	# return
	return df_train, df_valid, df_test

# define function for histogram
def CREATE_HISTOGRAM(list_data, str_filename='./output/plt_histogram.png', str_name='lgdtarget', tpl_figsize=(10,10), logger=None):
	# fig
	fig, ax = plt.subplots(figsize=tpl_figsize)
	# title
	ax.set_title(f'Distribution of {str_name}')
	# plot 
	sns.histplot(list_data, ax=ax, kde=True)
	# fix overlap
	fig.tight_layout()
	# save
	fig.savefig(str_filename, bbox_inches='tight')
	# log
	if logger:
		logger.warning(f'Saved plot to {str_filename}')
	# return
	return fig


# define function to find optimal n_components for PCA
def PLOT_PCA_EXPLAINED_VARIANCE(df, int_n_components_min=1, int_n_components_max=259,
								tpl_figsize=(12,10), str_filename='./output/plt_pca.png',
								logger=None):
	# list to append to
	list_flt_expl_var = []
	# iterate through n_components
	for n_components in range(int_n_components_min, int_n_components_max+1):
		# print status
		print(f'PCA - n_components {n_components}/{int_n_components_max}')
		# instantiate class
		cls_pca = PCA(n_components=n_components, 
				      copy=True, 
					  whiten=False, 
				      svd_solver='auto', 
				      tol=0.0, 
				      iterated_power='auto', 
				      random_state=42)
		# fit to df
		cls_pca.fit(df)
		# get explained variance
		flt_expl_var = np.sum(cls_pca.explained_variance_ratio_)
		# append to list
		list_flt_expl_var.append(flt_expl_var)
	# create empty canvas
	fig, ax = plt.subplots(figsize=tpl_figsize)
	# title
	ax.set_title('Explained Variance by n_components (PCA)')
	# x axis
	ax.set_xlabel('n_components')
	# y axis
	ax.set_ylabel('Explained Variance')
	# plot it
	ax.plot([item for item in range(int_n_components_min, int_n_components_max+1)] , list_flt_expl_var)
	# save fig
	plt.savefig(str_filename, bbox_inches='tight')
	# if using logging
	if logger:
		logger.warning(f'PCA explained variance plot generated and saved to {str_filename}')
	# return
	return fig
