# preprocessing
from .create_constants import CreateConstants
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import time
import datetime as dt
import pandas as pd
import numpy as np
from tqdm import tqdm

# define class
class Preprocessing(CreateConstants):
	# initialize
	def __init__(self, str_dirname_output='./output', str_target='target', str_datecol='date', bool_target_binary=True):
		# initialize parent class
		CreateConstants.__init__(self, str_dirname_output, str_target, str_datecol, bool_target_binary)
		# save arguments to object
		self.str_dirname_output = str_dirname_output
		self.str_target = str_target
		self.str_datecol = str_datecol
		self.bool_target_binary = bool_target_binary
	# show transformers
	def show_transformers(self, list_transformers):
		# iterate
		for a, transformer in enumerate(list_transformers):
			print(f'{a+1}: {transformer.__class__.__name__} - {transformer.str_message}')
		# return object
		return self

# data type setter
class SetDataTypes(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, bool_obj_to_string=True, bool_iterate=True, bool_verbose=True, str_message='Setting Dtypes'):
		self.bool_obj_to_string = bool_obj_to_string
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
		self.str_message = str_message
	# fit
	def fit(self, X, y=None):
		# get the data types into a dictionary
		dict_dtypes = dict(X.dtypes)
		# save to object
		self.dict_dtypes = dict_dtypes
		return self
	# transform
	def transform(self, X):
		# fillna
		time_start = time.perf_counter()

		# rm key val pairs not in X
		dict_dtypes = {key: val for key, val in self.dict_dtypes.items() if key in list(X.columns)}
		# if setting to string
		if self.bool_obj_to_string:
			# change O to str
			dict_dtypes = {key: ('str' if val == 'O' else val) for key, val in dict_dtypes.items()}
		# if iterating
		if self.bool_iterate:
			for key, val in tqdm (dict_dtypes.items()):
				X[key] = X[key].astype(val)
		# if not iterating
		else:
			X = X.astype(dict_dtypes)

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		if self.bool_verbose:
			print(f'Data Type Setter: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		# return
		return X

# class for cleaning text
class CleanText(BaseEstimator, TransformerMixin):
	# initialize class
	def __init__(self, list_cols, bool_iterate=True, bool_verbose=True, str_message='removing whitespace and lowering case and removing hyphens'):
		self.list_cols = list_cols
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
		self.str_message = str_message
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# start timer
		time_start = time.perf_counter()

		# define helper function
		def clean_text(str_text):
			# if NaN
			if not pd.isnull(str_text):
				# strip leading/trailing whitespace, rm spaces, lower
				return str(str_text).strip().replace(' ', '').replace('-','').lower()
			else:
				return str_text

		# future proof
		list_cols = [col for col in self.list_cols if col in list(X.columns)]
		# if iterating
		if self.bool_iterate:
			for col in tqdm (list_cols):
				X[col] = X[col].apply(clean_text)
		# if not iterating
		else:
			X[list_cols] = X[list_cols].applymap(lambda x: clean_text(str_text=x))

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')
		# return
		return X

# define class for cyclic features
class CyclicFeatures(BaseEstimator, TransformerMixin):
	# init
	def __init__(self, str_datecol='applicationdate', bool_verbose=True):
		self.str_datecol = str_datecol
		self.bool_verbose = bool_verbose
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# start timer
		time_start = time.perf_counter()

		# Quarter relative to year
		# sin
		X[f'ENG-{self.str_datecol}-quarter_year_sin'] = np.sin((X[self.str_datecol].dt.quarter-1) * (2*np.pi/4))
		# cos
		X[f'ENG-{self.str_datecol}-quarter_year_cos'] = np.cos((X[self.str_datecol].dt.quarter-1) * (2*np.pi/4))
		# tan
		X[f'ENG-{self.str_datecol}-quarter_year_tan'] = X[f'ENG-{self.str_datecol}-quarter_year_sin'] / X[f'ENG-{self.str_datecol}-quarter_year_cos']
		# Month relative to year
		# sin
		X[f'ENG-{self.str_datecol}-month_year_sin'] = np.sin((X[self.str_datecol].dt.month-1) * (2*np.pi/12))
		# cos
		X[f'ENG-{self.str_datecol}-month_year_cos'] = np.cos((X[self.str_datecol].dt.month-1) * (2*np.pi/12))
		# tan
		X[f'ENG-{self.str_datecol}-month_year_tan'] = X[f'ENG-{self.str_datecol}-month_year_sin'] / X[f'ENG-{self.str_datecol}-month_year_cos']
		# Day relative to week
		# sin
		X[f'ENG-{self.str_datecol}-day_week_sin'] = np.sin((X[self.str_datecol].dt.dayofweek-1) * (2*np.pi/7))
		# cos
		X[f'ENG-{self.str_datecol}-day_week_cos'] = np.cos((X[self.str_datecol].dt.dayofweek-1) * (2*np.pi/7))
		# tan
		X[f'ENG-{self.str_datecol}-day_week_tan'] = X[f'ENG-{self.str_datecol}-day_week_sin'] / X[f'ENG-{self.str_datecol}-day_week_cos']
		# Day relative to month
		# sin
		X[f'ENG-{self.str_datecol}-day_month_sin'] = np.sin((X[self.str_datecol].dt.day-1) * (2*np.pi/X[self.str_datecol].dt.daysinmonth))
		# cos
		X[f'ENG-{self.str_datecol}-day_month_cos'] = np.cos((X[self.str_datecol].dt.day-1) * (2*np.pi/X[self.str_datecol].dt.daysinmonth))
		# tan
		X[f'ENG-{self.str_datecol}-day_month_tan'] = X[f'ENG-{self.str_datecol}-day_month_sin'] / X[f'ENG-{self.str_datecol}-day_month_cos']
		# Day relative to year
		# sin
		X[f'ENG-{self.str_datecol}-day_year_sin'] = np.sin((X[self.str_datecol].dt.dayofyear-1) * (2*np.pi/365))
		# cos
		X[f'ENG-{self.str_datecol}-day_year_cos'] = np.cos((X[self.str_datecol].dt.dayofyear-1) * (2*np.pi/365))
		# tan
		X[f'ENG-{self.str_datecol}-day_year_tan'] = X[f'ENG-{self.str_datecol}-day_year_sin'] / X[f'ENG-{self.str_datecol}-day_year_cos']

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		if self.bool_verbose:
			print(f'Cyclic Features: {flt_sec:0.5} sec.')
		# return
		return X

# rounding binner
class RoundBinning(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, dict_round, bool_iterate=True, bool_verbose=True, str_message='round binning'):
		self.dict_round = dict_round
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
		self.str_message = str_message
	# fit
	def fit(self, X):
		return self
	# transform
	def transform(self, X):
		# start time
		time_start = time.perf_counter()

		# rm key val pairs not in X
		dict_round = {key: val for key, val in self.dict_round.items() if key in list(X.columns)}
		# if iterating
		if self.bool_iterate:
			for key, val in tqdm (dict_round.items()):
				X[key] = val * round(pd.to_numeric(X[key]) / val)
		# if not iterating
		else:
			# define helper to make lambda function shorter
			def round_it(col, dict_round):
				# get val
				val = self.dict_round[col.name]
				# return
				return val * round(pd.to_numeric(col) / val)
			# apply function
			X[list(dict_round.keys())] = X[list(dict_round.keys())].apply(lambda col: round_it(col=col, dict_round=dict_round), axis=0)

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		# return X
		return X

# generic feature engineering class
class GenericFeatureEngineering(BaseEstimator, TransformerMixin):
	# init
	def __init__(self, dict_fe, str_datecol='applicationdate__app', bool_verbose=True, str_message='generic FE (PTI)'):
		self.dict_fe = dict_fe
		self.str_datecol = str_datecol
		self.bool_verbose = bool_verbose
		self.str_message = str_message
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# start time
		time_start = time.perf_counter()

		# total attempted
		int_attempted = 0
		# total success
		int_success = 0
		# iterate through key val pairs
		for str_key, list_tpl in self.dict_fe.items():
			# iterate through list of tuples
			for tpl in list_tpl:
				# get numerator
				str_numerator = tpl[0]
				# get denominator
				str_denominator = tpl[1]
				# get new col name
				str_new_col = f'ENG-{str_numerator}-{str_key}-{str_denominator}'
				# add attempt
				int_attempted += 1
				# add success
				int_success += 1
				# get series
				try:
					# logic for numerator
					if str_numerator == self.str_datecol:
						# ser numerator
						ser_numerator = X[str_numerator].dt.year
					else:
						# ser numerator
						ser_numerator = X[str_numerator]
					# logic for denominator
					if str_denominator == self.str_datecol:
						# ser denominator
						ser_denominator = X[str_denominator].dt.year
					else:
						# ser denominator
						ser_denominator = X[str_denominator]
				except KeyError:
					# subtract success
					int_success -= 1
					# skip the rest of the loop
					continue
				
				# calculate
				if str_key == 'div':
					X[str_new_col] = ser_numerator / ser_denominator
				elif str_key == 'mult':
					X[str_new_col] = ser_numerator * ser_denominator
				elif str_key == 'add':
					X[str_new_col] = ser_numerator + ser_denominator
				elif str_key == 'sub':
					X[str_new_col] = ser_numerator - ser_denominator

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')
		# return
		return X

# custom range mapper
class CustomRangeMapper(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, dict_custom_range_map, bool_iterate=True, bool_verbose=True, str_message='custom ranges for certain features'):
		self.dict_custom_range_map = dict_custom_range_map
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
		self.str_message = str_message
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# map
		time_start = time.perf_counter()

		# rm key val pairs not in X
		dict_custom_range_map = {key: val for key, val in self.dict_custom_range_map.items() if key in list(X.columns)}
		# if iterating
		if self.bool_iterate:
			for key, val in tqdm (dict_custom_range_map.items()):
				X[key] = X[key].apply(val)
		# if not iterating
		else:
			X[list(dict_custom_range_map.keys())] = X.apply(dict_custom_range_map)

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		# return
		return X

# imputer
class Imputer(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, dict_imputations, bool_iterate=True, bool_verbose=True, str_message='Imputing with dict_imputations'):
		self.dict_imputations = dict_imputations
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
		self.str_message = str_message
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# fillna
		time_start = time.perf_counter()

		# rm key val pairs not in X
		dict_imputations = {key: val for key, val in self.dict_imputations.items() if key in list(X.columns)}
		# if iterating
		if self.bool_iterate:
			for key, val in tqdm (dict_imputations.items()):
				X[key] = X[key].fillna(val)
		# if not iterating
		else:
			X.fillna(dict_imputations, inplace=True)

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		# return
		return X

# define value replacer class
class FeatureValueReplacer(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, dict_value_replace, bool_iterate=True, bool_verbose=True, str_message='Feature Value Replacer'):
		self.dict_value_replace = dict_value_replace
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
		self.str_message = str_message
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# start time
		time_start = time.perf_counter()

		# rm key val pairs not in X
		dict_value_replace = {key: val for key, val in self.dict_value_replace.items() if key in list(X.columns)}
		# if iterating
		if self.bool_iterate:
			for key, val in tqdm (dict_value_replace.items()):
				X[key] = X[key].replace(val)
		# if not iterating
		else:
			X.replace(self.dict_value_replace, inplace=True)

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# return message
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		# return
		return X

# replace inf and -inf with NaN
class ReplaceInf(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, list_cols, bool_iterate=True, bool_verbose=True, str_message='Replace inf'):
		self.list_cols = list_cols
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
		self.str_message = str_message
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# start time
		time_start = time.perf_counter()

		# future proof
		list_cols = [col for col in self.list_cols if col in list(X.columns)]

		# if iterating
		if self.bool_iterate:
			for col in tqdm (list_cols):
				X[col] = X[col].replace([np.inf, -np.inf], np.nan)
		# if not iterating
		else:
			X[list_cols] = X[list_cols].replace([np.inf, -np.inf], np.nan)

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		if self.bool_verbose:
			print(f'Inf Replacer: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
				# message
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')
		# return X
		return X

# define preprocessing model class
class PreprocessingModel(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, list_transformers):
		self.list_transformers = list_transformers
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# start time
		time_start = time.perf_counter()

		# iterate through transformers
		for transformer in self.list_transformers:
			X = transformer.transform(X)

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		print(f'Preprocessing Model: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		# return
		return X

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

# define string converter
class StringConverter(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, list_cols, bool_verbose=True, str_message='String converter'):
		self.list_cols = list_cols
		self.bool_verbose = bool_verbose
		self.str_message = str_message
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# start time
		time_start = time.perf_counter()

		# make sure all cols are in X
		list_cols = [col for col in self.list_cols if col in list(X.columns)]
		# convert to string
		X[list_cols] = X[list_cols].applymap(str)

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		print(f'String Converter: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		# message
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')
		# return
		return X

# # new amt financed
# class NewAmountFinanced(BaseEstimator, TransformerMixin):
# 	# initialize
# 	def __init__(self):
# 		pass
# 	# fit
# 	def fit(self, X, y=None):
# 		return self
# 	# transform
# 	def transform(self, X):
# 		# start time
# 		time_start = time.perf_counter()

# 		# calculate
# 		X['ENG-fltamountfinanced__app'] = X['fltamountfinanced__app'] - X['fltgapinsurance__app'] - X['fltservicecontract__app']

# 		# end time
# 		time_end = time.perf_counter()
# 		# flt_sec
# 		flt_sec = time_end - time_start
# 		# print
# 		print(f'New Amount Financed: {flt_sec:0.5} sec.')
# 		# save to object
# 		self.flt_sec = flt_sec
# 		# return X
# 		return X

# class for dynamic feature engineering, dictionary key = str operation, value = list of lists of numerator and denominator
class DynamicFeatureEngineering(BaseEstimator, TransformerMixin):
	# init
	def __init__(self, dict_fe, bool_verbose=True, str_message='Dynamic Feature Engineering'):
		self.dict_fe = dict_fe
		self.str_message = str_message
		self.bool_verbose = bool_verbose

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		# start timer
		time_start = time.perf_counter()

		for key, list_list in self.dict_fe.items():
			for list_ in tqdm(list_list):
				str_numerator = list_[0]
				str_denominator = list_[1]
				str_new_col = f'ENG-{str_numerator}_{key}_{str_denominator}'
				
				if key == 'add':
					try:
						X[str_new_col] = X[str_numerator] + X[str_denominator]
					except:
						pass
				elif key =='subtract':
					try:
						X[str_new_col] = X[str_numerator] - X[str_denominator]
					except:
						pass
				elif key =='multiply':
					try:
						X[str_new_col] = X[str_numerator] * X[str_denominator]
					except:
						pass
				elif key =='divide':
					try:
						X[str_new_col] = X[str_numerator] / X[str_denominator]
					except:
						pass		

			# end time
			time_end = time.perf_counter()
			# time it took
			flt_sec = time_end - time_start

		# message
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')
		
		# return 
		return X 


# date parts 
class DatePartition(BaseEstimator, TransformerMixin):
	# initialize class
	def __init__(self,str_datecol='applicationdate__app', bool_verbose=True, str_message='Date partition'):
		self.str_datecol = str_datecol
		self.bool_verbose = bool_verbose
		self.str_message = str_message

	# fit
	def fit(self, X, y=None):
		return self

	# transform
	def transform(self, X):
		# start timer 
		time_start = time.perf_counter()

		# day of week
		X[f'ENG-{self.str_datecol}-dayofweek'] = X[self.str_datecol].dt.dayofweek

		# day of month
		X[f'ENG-{self.str_datecol}-month'] = X[self.str_datecol].dt.month

		# day of quarter
		X[f'ENG-{self.str_datecol}-quarter'] = X[self.str_datecol].dt.quarter

		# end time
		time_end = time.perf_counter()
		# flt sec
		flt_sec = time_end - time_start
		# print
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')

		# return
		return X 


# class for inflation
class Inflation(BaseEstimator, TransformerMixin):
	# initialize class
	def __init__(self, list_cols, dict_inflation_rate, dict_replace_errors, bool_iterate=True, bool_verbose=True, str_datecol='applicationdate__app', str_message='Inflator calculator', bool_rm_error_codes=True, bool_drop=True):
		self.list_cols = list_cols
		self.dict_inflation_rate = dict_inflation_rate
		self.dict_replace_errors = dict_replace_errors
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
		self.str_datecol = str_datecol
		self.str_message = str_message
		self.bool_rm_error_codes = bool_rm_error_codes
		self.bool_drop = bool_drop

	# fit
	def fit (self, X, y=None):
		return self

	# transform
	def transform(self, X):
		# start timer
		time_start = time.perf_counter()

		# future proof
		list_cols = [col for col in self.list_cols if col in list(X.columns)]

		# create year
		X['year'] = X[self.str_datecol].dt.year
		# map factor to year
		X['factor'] = X['year'].map(self.dict_inflation_rate)

		# convert
		if self.bool_iterate:
			for col in tqdm(list_cols):
				# multiply by factor
				X[col] = X[col]* X['factor']
		else:
			X[list_cols] = X[list_cols].multiply(X['factor'], axis='index')

		# replaec error codes
		if self.bool_rm_error_codes:
			# replace
			X[list_cols] = X[list_cols].replace(self.dict_replace_errors, inplace=False)
		else:
			pass 

		# drop
		if self.bool_drop:
			X = X.drop(['year', 'factor'], axis=1, inplace=False)

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')

		# return 
		return X 



# clip values
class ValueClipper(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, list_cols, bool_iterate=True, bool_verbose=True, str_message='Value clipper', int_clip=0):
		self.list_cols = list_cols
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
		self.str_message = str_message
		self.int_clip = int_clip
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# start time
		time_start = time.perf_counter()

		# future proof
		list_cols = [col for col in self.list_cols if col in list(X.columns)]

		# if iterating
		if self.bool_iterate:
			for col in tqdm (list_cols):
				X[col] = np.clip(a=X[col], a_min=self.int_clip, a_max=None)
		else:
			X[list_cols] = np.clip(a=X[list_cols], a_min=self.int_clip, a_max=None)

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		# return X
		return X


class ForwardFillByAccount(BaseEstimator, TransformerMixin):
    # initialize
    def __init__(self, list_cols, bool_verbose=True, str_message='Forward fill'):
        self.list_cols = list_cols
        self.bool_verbose = bool_verbose
        self.str_message = str_message
    # fit
    def fit(self, X, y=None):
        return self
    # transform
    def transform(self, X):
        # fillna
        time_start = time.perf_counter()
        
        # future-proof
        list_cols = [col for col in self.list_cols if col in list(X.columns)]
        
        # forward fill
        for col in tqdm (list_cols):
        	X[list_cols] = X.groupby('bigAccountid__base', as_index=False)[list_cols].transform(
            	lambda x: x.ffill(),
        	)
        
        # end time
        time_end = time.perf_counter()
        # flt_sec
        flt_sec = time_end - time_start
        # print
        if self.bool_verbose:
            print(f'{self.str_message}: {flt_sec:0.5} sec.')
        # save to object
        self.flt_sec = flt_sec
        # return
        return X 


class FillNaNByAccount(BaseEstimator, TransformerMixin):
    # initialize
    def __init__(self, list_cols, bool_verbose=True, str_message='Fill NaN by account'):
        self.list_cols = list_cols
        self.bool_verbose = bool_verbose
        self.str_message = str_message
    # fit
    def fit(self, X, y=None):
        # get min of each col
        dict_imputations = {}
        for col in self.list_cols:
            # get min
            val_impute = np.min(X[col])
            # assign
            dict_imputations[col] = val_impute
        # save to object
        self.dict_imputations = dict_imputations
        # return object
        return self
    # transform
    def transform(self, X):
        # fillna
        time_start = time.perf_counter()
        
        # future-proof
        list_cols = [col for col in self.list_cols if col in list(X.columns)]
        
        # fill na by account and then fill the rest with min
        for col in tqdm (list_cols):
            X[col] = X.groupby('bigAccountid__base', as_index=False)[col].transform(
                lambda x: x.fillna(np.mean(x)),
            )
            # get imputation val
            val_impute = self.dict_imputations[col]
            # impute
            X[col] = X[col].fillna(val_impute)
        
        # end time
        time_end = time.perf_counter()
        # flt_sec
        flt_sec = time_end - time_start
        # print
        if self.bool_verbose:
            print(f'{self.str_message}: {flt_sec:0.5} sec.')
        # save to object
        self.flt_sec = flt_sec
        # return
        return X


class FillNaNByAccountZero(BaseEstimator, TransformerMixin):
    # initialize
    def __init__(self, list_cols, bool_verbose=True, str_message='Fill NaN by account'):
        self.list_cols = list_cols
        self.bool_verbose = bool_verbose
        self.str_message = str_message
    # fit
    def fit(self, X, y=None):
        # get min of each col
        dict_imputations = {}
        for col in self.list_cols:
            # get min
            val_impute = np.min(X[col])
            # assign
            dict_imputations[col] = val_impute
        # save to object
        self.dict_imputations = dict_imputations
        # return object
        return self
    # transform
    def transform(self, X):
        # fillna
        time_start = time.perf_counter()
        
        # future-proof
        list_cols = [col for col in self.list_cols if col in list(X.columns)]
        
        # fill na by account and then fill the rest with min
        for col in tqdm (list_cols):
            X[col] = X.groupby('bigAccountid__base', as_index=False)[col].transform(
                lambda x: x.fillna(0),
            )
            # get imputation val
            val_impute = self.dict_imputations[col]
            # impute
            X[col] = X[col].fillna(val_impute)
        
        # end time
        time_end = time.perf_counter()
        # flt_sec
        flt_sec = time_end - time_start
        # print
        if self.bool_verbose:
            print(f'{self.str_message}: {flt_sec:0.5} sec.')
        # save to object
        self.flt_sec = flt_sec
        # return
        return X


class DynamicFeatureEngineeringSum(BaseEstimator, TransformerMixin):
	# init
	def __init__(self, dict_fe_add, bool_verbose=True, str_message='Dynamic Feature Engineering'):
		self.dict_fe_add = dict_fe_add
		self.str_message = str_message
		self.bool_verbose = bool_verbose

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		# start timer
		time_start = time.perf_counter()

		for key, list_list in self.dict_fe_add.items():
			for list_ in tqdm(list_list):
				new_list = []
				for s in list_:
					new_list.append(s[:-16])
				str_name = '_'.join(new_list)
				str_new_col = f'ENG-{str_name}_sum'
				if key == 'add':
					try:
						X[str_new_col] = X[list_].sum(axis=1)
					except:
						pass	

			# end time
			time_end = time.perf_counter()
			# time it took
			flt_sec = time_end - time_start

		# message
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')
		
		# return 
		return X 


class DynamicFeatureEngineeringRetention(BaseEstimator, TransformerMixin):
	# init
	def __init__(self, dict_fe, bool_verbose=True, str_message='Dynamic Feature Engineering'):
		self.dict_fe = dict_fe
		self.str_message = str_message
		self.bool_verbose = bool_verbose

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		# start timer
		time_start = time.perf_counter()

		for key, list_list in self.dict_fe.items():
			for list_ in tqdm(list_list):
				str_numerator = list_[0]
				str_denominator = list_[1]
				str_new_col = f'ENG-{str_numerator[:-16]}_{key}_{str_denominator[:-16]}_{str_numerator[-16:]}'
				
				if key == 'add':
					try:
						X[str_new_col] = X[str_numerator] + X[str_denominator]
					except:
						pass
				elif key =='subtract':
					try:
						X[str_new_col] = X[str_numerator] - X[str_denominator]
					except:
						pass
				elif key =='multiply':
					try:
						X[str_new_col] = X[str_numerator] * X[str_denominator]
					except:
						pass
				elif key =='divide':
					try:
						X[str_new_col] = X[str_numerator] / X[str_denominator]
					except:
						pass		

			# end time
			time_end = time.perf_counter()
			# time it took
			flt_sec = time_end - time_start

		# message
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')
		# return 
		return X 


# imputer
class ImputerRetention(BaseEstimator, TransformerMixin):
    # initialize
    def __init__(self, dict_imputations, bool_verbose=True, str_message='Imputer'):
        self.dict_imputations = dict_imputations
        self.bool_verbose = bool_verbose
        self.str_message = str_message
    # fit
    def fit(self, X, y=None):
        return self
    # transform
    def transform(self, X):
        # fillna
        time_start = time.perf_counter()

        # rm key val pairs not in X
        dict_imputations = {key: val for key, val in self.dict_imputations.items() if key in list(X.columns)}
        
        # iterate
        for key, val in tqdm (dict_imputations.items()):
            X[key] = X[key].fillna(val)
        
        # end time
        time_end = time.perf_counter()
        # flt_sec
        flt_sec = time_end - time_start
        # print
        if self.bool_verbose:
            print(f'{self.str_message}: {flt_sec:0.5} sec.')
        # save to object
        self.flt_sec = flt_sec
        # return
        return X   


class MilesPerYear(BaseEstimator, TransformerMixin):
    # initialize
    def __init__(self, bool_verbose=True, str_message='Vehicle age'):
        self.bool_verbose = bool_verbose
        self.str_message = str_message
    # fit
    def fit(self, X, y=None):
        return self
    # transform
    def transform(self, X):
        # fillna
        time_start = time.perf_counter()
        
        # vehicle age
        this_year = dt.date.today().year
        X['ENG-vehicle_age'] = this_year - X['VehicleYear__tsp'] # convert months to years
        X['ENG-miles_per_year'] = X['Miles_Odometer__tsp'] / X['ENG-vehicle_age'] # # add years to funding age

        # end time
        time_end = time.perf_counter()
        # flt_sec
        flt_sec = time_end - time_start
        # print
        if self.bool_verbose:
            print(f'{self.str_message}: {flt_sec:0.5} sec.')
        # save to object
        self.flt_sec = flt_sec
        # return
        return X  


class StructuredFeatureEngineering(BaseEstimator, TransformerMixin):
    # initialize
    def __init__(self, bool_verbose=True, str_message='Feature engineering for structured features'):
        self.bool_verbose = bool_verbose
        self.str_message = str_message
    # fit
    def fit(self, X, y=None):
        return self
    # transform
    def transform(self, X):

    	# time start
        time_start = time.perf_counter()
        
        X['ENG-down_to_loan'] = X['fltDownCash__tsp'] / X['AmtFinanced__tsp']
        X['ENG-adv_to_loan'] = X['fltAdvance__tsp'] / X['AmtFinanced__tsp']
        X['ENG-buydown_to_loan'] = X['fltBuyDownFee__tsp'] / X['AmtFinanced__tsp']
        X['ENG-acq_to_loan'] = X['fltAcquisitionFee__tsp'] / X['AmtFinanced__tsp']
        X['ENG-pmt_to_loan'] = X['fltPaymentOriginal__tsp'] / X['AmtFinanced__tsp']
        X['ENG-down_to_income'] = X['fltDownCash__tsp'] / X['income__tsp']

        # end time
        time_end = time.perf_counter()
        # flt_sec
        flt_sec = time_end - time_start
        # print
        if self.bool_verbose:
            print(f'{self.str_message}: {flt_sec:0.5} sec.')
        # save to object
        self.flt_sec = flt_sec
        # return
        return X  


class VarianceComparator(BaseEstimator, TransformerMixin):
    def __init__(self, list_tuples, bool_verbose=True, str_message='Checking to see if port review FE old/new are the same'):
        self.list_tuples = list_tuples
        self.bool_verbose = bool_verbose
        self.str_message = str_message

    # fit
    def fit(self, X, y=None):
    	return self

    # transform
    def transform(self, X):
    	# time start
        time_start = time.perf_counter()

        for column1, column2 in self.list_tuples:
            # Compute the variances of the columns
            var1 = X[column1].var()
            var2 = X[column2].var()

            # Drop the column with no variance
            if var1 == var2:
                X.drop(columns=[column1], inplace=True)

        # end time
        time_end = time.perf_counter()
        # flt_sec
        flt_sec = time_end - time_start
        # print
        if self.bool_verbose:
            print(f'{self.str_message}: {flt_sec:0.5} sec.')
        # save to object
        self.flt_sec = flt_sec

        return X

		


'''
to use previous class
1. call the current class
	cls_current = current_module(arguments=arguments)

2. call previous module from current class, assign to new object perhaps with previous module name
	cls_previous = cls_current.previous_module(argument=argument...)

3. make sure to review arguments from previous module in order to update:
	file name, path, etc. 


'''

class VarianceComparatorBoth(BaseEstimator, TransformerMixin):
    def __init__(self, list_tuples, bool_verbose=True, str_message='Checking to see if port review FE old/new are the same'):
        self.list_tuples = list_tuples
        self.bool_verbose = bool_verbose
        self.str_message = str_message

    # fit
    def fit(self, X, y=None):
    	return self

    # transform
    def transform(self, X):
    	# time start
        time_start = time.perf_counter()

        for column1, column2 in self.list_tuples:
            # Compute the variances of the columns
            var1 = X[column1].var()
            var2 = X[column2].var()

            # Drop the column with no variance
            if var1 == 0 and var2 == 0:
                X.drop(columns=[column1], inplace=True)
                X.drop(columns=[column2], inplace=True)

        # end time
        time_end = time.perf_counter()
        # flt_sec
        flt_sec = time_end - time_start
        # print
        if self.bool_verbose:
            print(f'{self.str_message}: {flt_sec:0.5} sec.')
        # save to object
        self.flt_sec = flt_sec
        return X

class DropLeakyDate(BaseEstimator, TransformerMixin):
	def __init__(self, list_leaky_dates_drop, bool_verbose=True, str_message='dropping default date and termination date'):
		self.list_leaky_dates_drop=list_leaky_dates_drop
		self.bool_verbose=bool_verbose,
		self.str_message=str_message

	#fit
	def fit(self, X, y=None):
		return self

	# transform
	def transform(self, X):
		# time start
		time_start = time.perf_counter()
		for i in tqdm (self.list_leaky_dates_drop):
			try:
				X.drop(columns=[i], inplace=True)
			except:
				pass
		# end time
		time_end = time.perf_counter()
		# flt_sec 
		flt_sec = time_end - time_start
		# print
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		return X 

