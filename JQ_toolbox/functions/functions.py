# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:48:31 2021

@author: jqiu
"""
import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import logging

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

# define function to read csv
def READ_CSV(str_filename='D:/output_data/df_raw.csv', list_usecols=None, int_nrows=None, logger=None):
	#start timer
	time_start = time.perf_counter()
	#read csv
	df = pd.read_csv(str_filename, nrows=int_nrows, usecols=list_usecols)
	#log it
	if logger:
		logger.warning(f'Data imported from csv in {(time.perf_counter()-time_start)/60:0.4} min.')
	# return
	return df

# define class for binning by rounding to nearest number
class RoundBinner(BaseEstimator, TransformerMixin):
	# init
	def __init__(self, dict_round):
		self.dict_round = dict_round
	# fit 
	def fit(self, X):
		return self
	# transform
	def transform(self, X):
		time_start = time.perf_counter()
		# make copy of dict_round
		dict_round = self.dict_round.copy()
		# get list of keys
		list_keys = list(dict_round.keys())
		# iterate through the keys
		for key in list_keys:
			#if key is not in data frame
			if key not in list(X.columns):
				#delete key from dictionary
				del dict_round[key]
		# iterate through dictionary
		for key, val in dict_round.items():
			X[key] = val * round(pd.to_numeric(X[key]) / val)
		print(f'Time to bin: {time.perf_counter()-time_start:0.5} sec.')
		# return X 
		return X 

# define class for quantile binning 
