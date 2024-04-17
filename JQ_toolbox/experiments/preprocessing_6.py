# preprocessing
from .create_constants_5 import CreateConstants
from sklearn.base import BaseEstimator, TransformerMixin
import time
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
			print(f'{a+1}: {transformer.__class__.__name__}')
		# return object
		return self

# data type setter
class SetDataTypes(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, bool_obj_to_string=True, bool_iterate=True, bool_verbose=True):
		self.bool_obj_to_string = bool_obj_to_string
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
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
	def __init__(self, list_cols, bool_iterate=True, bool_verbose=True):
		self.list_cols = list_cols
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
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
				return str(str_text).strip().replace(' ', '').lower()
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
			print(f'Text Cleaner: {flt_sec:0.5} sec.')
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
	def __init__(self, dict_round, bool_iterate=True, bool_verbose=True):
		self.dict_round = dict_round
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
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
			print(f'Binner: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		# return X
		return X

# generic feature engineering class
class GenericFeatureEngineering(BaseEstimator, TransformerMixin):
	# init
	def __init__(self, dict_fe, str_datecol='applicationdate__app', bool_verbose=True):
		self.dict_fe = dict_fe
		self.str_datecol = str_datecol
		self.bool_verbose = bool_verbose
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
			print(f'Feature Engineer ({int_success}/{int_attempted}): {flt_sec:0.5} sec.')
		# return
		return X

# custom range mapper
class CustomRangeMapper(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, dict_custom_range_map, bool_iterate=True, bool_verbose=True):
		self.dict_custom_range_map = dict_custom_range_map
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
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
			print(f'Custom Range Mapper: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		# return
		return X

# imputer
class Imputer(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, dict_imputations, bool_iterate=True, bool_verbose=True):
		self.dict_imputations = dict_imputations
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
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
			print(f'Imputer: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		# return
		return X

# define value replacer class
class FeatureValueReplacer(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, dict_value_replace, bool_iterate=True, bool_verbose=True):
		self.dict_value_replace = dict_value_replace
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
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
		# print
		if self.bool_verbose:
			print(f'Value Replacer: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		# return
		return X

# replace inf and -inf with NaN
class ReplaceInf(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, list_cols, bool_iterate=True, bool_verbose=True):
		self.list_cols = list_cols
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
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
	def __init__(self, list_cols):
		self.list_cols = list_cols
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
		# return
		return X

# new amt financed
class NewAmountFinanced(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self):
		pass
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# start time
		time_start = time.perf_counter()

		# calculate
		X['ENG-fltamountfinanced__app'] = X['fltamountfinanced__app'] - X['fltgapinsurance__app'] - X['fltservicecontract__app']

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		print(f'New Amount Financed: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		# return X
		return X


# class for dynamic feature engineering, dictionary key = str operation, value = list of lists of numerator and denominator
class DynamicFE(BaseEstimator, TransformerMixin):
	# init
	def __init__(self, dict_fe):
		self.dict_fe = dict_fe

	def fit(self, X, y=None):
		return X

	def transform(self, X):
		for key, list_list in self.dict_fe.items():
			for list_ in list_list:
				str_numerator = list_[0]
				str_denominator = list_[1]
				str_new_col = f'eng_{str_numerator}_{key}_{str_denominator}'
				
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
		return X 