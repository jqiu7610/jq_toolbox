import logging
import os
import time
import pandas as pd
import pickle
from pandas.api.types import is_numeric_dtype
import json
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import datetime as dt
import git


#===============================================================================
# CLASSES
#===============================================================================
# make class for csv to df
class CSVRelated():
	# init
	def __init__(self, logger=None):
		self.logger = logger

	# function for loading in csv
	def csv_to_df(self, str_filename='../output_data/df_raw.csv', list_usecols=None, list_parse_dates=None, int_nrows=None,
	          list_skiprows=None, str_sep=',', lambda_date_parser=None, str_encoding=None):
		# start timer
		time_start = time.perf_counter()
		# read json file
		df = pd.read_csv(str_filename, parse_dates=list_parse_dates, usecols=list_usecols, nrows=int_nrows, 
			             skiprows=list_skiprows, sep=str_sep, date_parser=lambda_date_parser,
			             encoding=str_encoding)
		self.df = df 
		# if we are using a logger
		if self.logger:
		# log it
			self.logger.warning(f'Data imported from {str_filename} in {(time.perf_counter()-time_start)/60:0.4} min.')
		# message for terminal
		print(f'Data imported from {str_filename} in {(time.perf_counter()-time_start)/60:0.4} min.')
		# return
		return self
	
	# function for df to csv no index
	def df_to_csv(self, df, str_filename='./output/df_experiment.csv'):
		df.to_csv(str_filename, index=False)
		# start timer
		time_start = time.perf_counter()
		# if logger
		if self.logger:
		# log it
			self.logger.warning(f'DF converted to {str_filename} in {(time.perf_counter()-time_start)/60:0.4} min.')
		# message for terminal
		print(f'DF converted to {str_filename} in {(time.perf_counter()-time_start)/60:0.4} min.')
		# return
		return self


# make pickle functions into class
class PickleRelated():
	# init
	def __init__(self, logger=None):
		self.logger = logger

	#  function for pickle loading
	def load_pickle(self, str_filename):
		# get file
		pickled_file = pickle.load(open(str_filename, 'rb'))
		# set to self
		self.pickled_file = pickled_file

		# if using logger
		if self.logger:
 			# log it
 			self.logger.warning(f'Imported file from {str_filename}')
		# return 
		return self

	# function for pickle dumping
	def pickle_object(self, item_to_pickle, str_filename):
		# pickle file
		pickle.dump(item_to_pickle, open(str_filename, 'wb'))
		# if using logger
		if self.logger:
		# log it
			self.logger.warning(f'Pickled {item_to_pickle.__class__.__name__} to {str_filename}')
		return self





#===============================================================================
# FUNCTIONS
#===============================================================================
# define function for logging
def LOG_EVENTS(str_filename='./logs/db_pull.log'):
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


# define function for loadin#g from pickle
def LOAD_FROM_PICKLE(logger=None, str_filename='../06_preprocessing/output/dict_imputations.pkl'):
	# get file
	pickled_file = pickle.load(open(str_filename, 'rb'))
	# if using logger
	if logger:
		# log it
		logger.warning(f'Imported file from {str_filename}')
	# return
	return pickled_file


# define function to read csv
def CSV_TO_DF(logger=None, str_filename='../output_data/df_raw.csv', list_usecols=None, list_parse_dates=None, int_nrows=None,
	          list_skiprows=None, str_sep=',', lambda_date_parser=None, str_encoding=None):
	# start timer
	time_start = time.perf_counter()
	# read json file
	df = pd.read_csv(str_filename, parse_dates=list_parse_dates, usecols=list_usecols, nrows=int_nrows, 
		             skiprows=list_skiprows, sep=str_sep, date_parser=lambda_date_parser,
		             encoding=str_encoding)
	# if we are using a logger
	if logger:
		# log it
		logger.warning(f'Data imported from {str_filename} in {(time.perf_counter()-time_start)/60:0.4} min.')
	# return
	return df


# define function to get numeric and non-numeric cols
def GET_NUMERIC_AND_NONNUMERIC(df, list_columns, logger=None):
	# instantiate empty lists
	list_numeric = []
	list_non_numeric = []
	# iterate through list_columns
	for col in list_columns:
		# if its numeric
		if is_numeric_dtype(df[col]):
			# append to list_numeric
			list_numeric.append(col)
		else:
			# append to list_non_numeric
			list_non_numeric.append(col)
	# if using logger
	if logger:
		logger.warning(f'{len(list_numeric)} numeric columns identified, {len(list_non_numeric)} non-numeric columns identified')
	# print message
	print(f'{len(list_numeric)} numeric columns identified, {len(list_non_numeric)} non-numeric columns identified')
	# return both lists
	return list_numeric, list_non_numeric

# define function for writing to pickle
def PICKLE_TO_FILE(item_to_pickle, str_filename='./output/transformer.pkl', logger=None):
	# pickle file
	pickle.dump(item_to_pickle, open(str_filename, 'wb'))
	# if using logger
	if logger:
		# log it
		logger.warning(f'Pickled {item_to_pickle.__class__.__name__} to {str_filename}')
	# print message
	print(f'Pickled {item_to_pickle.__class__.__name__} to {str_filename}')

# write dictionary to text
def DICT_TO_TEXT(dict_, str_filename='./output/dict_evalmetrics.txt', logger=None):
	# write dictionary to text
	with open(str_filename, 'w') as file:
		file.write(json.dumps(dict_))
	# if using logger
	if logger:
		logger.warning(f'Wrote dictionary to {str_filename}')
	# print
	print(f'Wrote dictionary to {str_filename}')


# define function 
def SORT_DF(df, str_colname='dtmStampCreation__app', logger=None, bool_dropcol=False):
	# get series of str_colname
	ser_ = df[str_colname]
	# sort ascending
	ser_sorted = ser_.sort_values(ascending=True)
	# get the index as a list
	list_ser_sorted_index = list(ser_sorted.index)
	# order df
	df = df.reindex(list_ser_sorted_index)
	# if dropping str_colname
	if bool_dropcol:
		#drop str_colname
		del df[str_colname]
	# if using a logger
	if logger:
		logger.warning(f'df sorted ascending by {str_colname}.')
	# print message
	print(f'df sorted ascending by {str_colname}.')
	# return
	return df


# define function for splitting into X and y
def X_Y_SPLIT(df_train, df_valid, logger=None, str_targetname='TARGET__app'):
	# train
	y_train = df_train[str_targetname]
	df_train.drop(str_targetname, axis=1, inplace=True)
	# valid
	y_valid = df_valid[str_targetname]
	df_valid.drop(str_targetname, axis=1, inplace=True)
	# if using logger
	if logger:
		# log it
		logger.warning('Train and valid dfs split into X and y')
	# return
	return df_train, y_train, df_valid, y_valid

# define function for loadin#g from pickle
def LOAD_FROM_PICKLE(logger=None, str_filename='../06_preprocessing/output/dict_imputations.pkl'):
	# get file
	pickled_file = pickle.load(open(str_filename, 'rb'))
	# if using logger
	if logger:
		# log it
		logger.warning(f'Imported file from {str_filename}')
	# return
	return pickled_file