from .general_1 import General
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
# import seaborn as sns
from tqdm import tqdm

# define EDA class that will inherit from general class

class Exploratory_Data_Analysis(General):
	# initialize
	def __init__(self, str_dirname_output='./output', str_target='target', str_datecol='date', bool_target_binary=True):
		# initialize parent class
		General.__init__(self, str_dirname_output)
		self.str_dirname_output = str_dirname_output
		self.str_target = str_target
		self.str_datecol = str_datecol
		self.bool_target_binary = bool_target_binary

	# get df info
	def get_df_info(self, df, str_filename='dict_df_info.json'):
		# nrows/ncols
		int_nrows, int_ncols = df.shape
		# total obs
		int_obs_total = int_nrows * int_ncols
		# tot na
		int_n_missing = np.sum(df.isnull().sum())
		# create dict
		dict_df_info = {
			'int_nrows': int_nrows,
			'int_ncols': int_ncols,
			'int_obs_total': int_obs_total,
			'date_min': str(np.min(df[self.str_datecol])),
			'date_max': str(np.max(df[self.str_datecol])),
			'flt_mean_target': np.mean(df[self.str_target]),
			'flt_propna': int_n_missing / int_obs_total,
		}
		# write to .json
		json.dump(dict_df_info, open(f'{self.str_dirname_output}/{str_filename}', 'w'))
		# save to object
		self.dict_df_info = dict_df_info
		# return object
		return self

	# get proportion NaN by column
	def get_prop_nan_by_column(self, df, str_filename='df_propna.csv'):
		# create df
		df_propna = pd.DataFrame(df.isnull().mean())
		# cols
		df_propna.columns = ['prop_na']
		# sort
		df_propna.sort_values(by='prop_na', ascending=False, inplace=True)
		# write to csv
		df_propna.to_csv(f'{self.str_dirname_output}/{str_filename}', index=True)
		# return self
		return self

	# get proportion NaN by ID columns
	def get_prop_nan_by_id_columns(self, df, list_cols_id, str_filename='df_propna_ids.csv'):
		# create df
		df_propna = pd.DataFrame(df[list_cols_id].isnull().mean())
		# cols
		df_propna.columns = ['prop_na']
		# sort
		df_propna.sort_values(by='prop_na', ascending=False, inplace=True)
		# write to csv
		df_propna.to_csv(f'{self.str_dirname_output}/{str_filename}', index=True)
		# return object
		return self

	# plot proportion NaN overall
	def plot_proportion_nan(self, df, str_filename='plt_prop_nan.png'):
		# get int_n_missing
		int_n_missing = np.sum(df.isnull().sum())
		# get int_obs_total
		int_obs_total = df.shape[0] * df.shape[1]
		# create axis
		fig, ax = plt.subplots(figsize=(10,15))
		# title
		ax.set_title('Pie Chart of Missing Values')
		ax.pie(
			x=[int_n_missing, int_obs_total], 
			colors=['y', 'c'],
			explode=(0, 0.1),
			labels=['Missing', 'Non-Missing'], 
			autopct='%1.1f%%',
		)
		# save fig
		plt.savefig(f'{self.str_dirname_output}/{str_filename}', bbox_inches='tight')
		# show plot
		plt.show()
		# close plot
		plt.close()
		# return object
		return self
	
	# plot data type frequency
	def plot_data_type_frequency(self, df, str_filename='plt_dtype.png'):
		# get numeric
		list_cols_numeric = [col for col in df.columns if is_numeric_dtype(df[col])]
		# get non-numeric
		list_cols_non_numeric = [col for col in df.columns if col not in list_cols_numeric]
		# get number of columns
		int_ncols = df.shape[1]
		# % numeric
		flt_pct_numeric = (len(list_cols_numeric) / int_ncols) * 100
		# % non-numeric
		flt_pct_non_numeric = (len(list_cols_non_numeric) / int_ncols) * 100
		# create ax
		fig, ax = plt.subplots(figsize=(10,10))
		# title
		ax.set_title(f'{flt_pct_numeric:0.4}% Numeric, {flt_pct_non_numeric:0.4}% Non-Numeric (N = {int_ncols})')
		# y label
		ax.set_ylabel('Frequency')
		# bar plot
		ax.bar(['Numeric','Non-Numeric'], [len(list_cols_numeric), len(list_cols_non_numeric)])
		# save plot
		plt.savefig(f'{self.str_dirname_output}/{str_filename}', bbox_inches='tight')
		# show plot
		plt.show()
		# close plot
		plt.close()
		# return object
		return self
	
	# # plot target
	# def plot_target(self, df, str_filename='plt_target.png'):
	# 	# if we have a binary target
	# 	if self.bool_target_binary:
	# 		# get the total positive
	# 		int_tot_pos = np.sum(df[self.str_target])
	# 		# get total
	# 		int_total = len(df[self.str_target])
	# 		# get the toeal negative
	# 		int_tot_neg = int_total - int_tot_pos
	# 		# get pct negative class
	# 		flt_pct_negative = (int_tot_neg / int_total) * 100
	# 		# get pct positive class
	# 		flt_pct_positive = (int_tot_pos / int_total) * 100
	# 		# create axis
	# 		fig, ax = plt.subplots(figsize=(15,10))
	# 		# title
	# 		ax.set_title(f'{flt_pct_negative:0.4}% = 0, {flt_pct_positive:0.4}% = 1, (N = {int_total})')
	# 		# frequency bar plot
	# 		ax.bar([0, 1], [int_tot_neg, int_tot_pos])
	# 		# ylabel
	# 		ax.set_ylabel('Frequency')
	# 		# xticks
	# 		ax.set_xticks([0, 1])
	# 		# xtick labels
	# 		ax.set_xticklabels(['0','1'])
		
	# 	# if we have a continuous target
	# 	else:
	# 		# fig
	# 		fig, ax = plt.subplots(figsize=(10,7))
	# 		# title
	# 		ax.set_title(f'Distribution of {self.str_target}')
	# 		# plot
	# 		sns.histplot(df[self.str_target], ax=ax, kde=True)
	# 	# fix overlap
	# 	fig.tight_layout()
	# 	# save
	# 	plt.savefig(f'{self.str_dirname_output}/{str_filename}', bbox_inches='tight')
	# 	# show plot
	# 	plt.show()
	# 	# close plot
	# 	plt.close()
	# 	# return object
	# 	return self

		# df info summary table
	def get_df_info_summary_table(self, dict_df_info_train, dict_df_info_valid, dict_df_info_test):
		# create df
		df_info_summary = pd.DataFrame({
			'Data Set': ['Train','Valid','Test'],
			'Rows': [dict_df_info_train['int_nrows'], dict_df_info_valid['int_nrows'], dict_df_info_test['int_nrows']],
			'Columns': [dict_df_info_train['int_ncols'], dict_df_info_valid['int_ncols'], dict_df_info_test['int_ncols']],
			'Total Obs.': [dict_df_info_train['int_obs_total'], dict_df_info_valid['int_obs_total'], dict_df_info_test['int_obs_total']],
			'Min. Date': [dict_df_info_train['date_min'], dict_df_info_valid['date_min'], dict_df_info_test['date_min']],
			'Max. Date': [dict_df_info_train['date_max'], dict_df_info_valid['date_max'], dict_df_info_test['date_max']],
			'Prop. NaN': [dict_df_info_train['flt_propna'], dict_df_info_valid['flt_propna'], dict_df_info_test['flt_propna']],
			'Target Mean': [dict_df_info_train['flt_mean_target'], dict_df_info_valid['flt_mean_target'], dict_df_info_test['flt_mean_target']],
		})
		# save
		df_info_summary.to_csv(f'{self.str_dirname_output}/df_info_summary.csv', index=False)
		# return object
		return self

		# get descriptives for each column
	def get_descriptives_by_column(self, df, str_filename='df_descriptives.csv'):
		# get descriptives
		list_dict_row = []
		for col in tqdm (df.columns):
			# save as series
			ser_col = df[col]
			# get proportion nan
			flt_prop_na = ser_col.isnull().mean()
			if flt_prop_na == 1.0:
				# create row
				dict_row = {
					'feature': col,
					'dtype': np.nan,
					'propna': flt_prop_na,
					'min': np.nan,
					'max': np.nan,
					'range': np.nan,
					'std': np.nan,
					'mean': np.nan,
					'median': np.nan,
					'mode': np.nan,
					'n_unique': 0,
					'prop_unique': 0,
					'prop_negative': np.nan,
					'prop_min': np.nan,
					'prop_max': np.nan,
					'prop_zero': np.nan,
					'mean_target_25': np.nan,
					'mean_target_50': np.nan,
					'mean_target_75': np.nan,
				}
				# append
				list_dict_row.append(dict_row)
				# skip the rest of the iteration
				continue
			# get data type
			str_dtype = ser_col.dtype
			# if value
			if str_dtype in ['float64', 'int64']:
				val_min, val_max, val_mean, val_std, val_median = ser_col.min(), ser_col.max(), ser_col.mean(), ser_col.std(), ser_col.median()
				val_range = val_max - val_min
				val_mode, int_n_unique = ser_col.mode().iloc[0], ser_col.nunique()
				flt_prop_unique = int_n_unique / len(ser_col.dropna())
				flt_prop_negative = len(ser_col[ser_col<0]) / len(ser_col.dropna())
				flt_prop_min = len(ser_col[ser_col==val_min]) / len(ser_col.dropna())
				flt_prop_max = len(ser_col[ser_col==val_max]) / len(ser_col.dropna())
				flt_prop_zero = len(ser_col[ser_col==0]) / len(ser_col.dropna())
				# get quantiles
				ser_quantiles = ser_col.quantile([0.25, 0.50, 0.75])
				df_tmp = df[[col, self.str_target]]
				flt_mean_target_25 = df_tmp[df_tmp[col]<=ser_quantiles[0.25]][self.str_target].mean()
				flt_mean_target_50 = df_tmp[df_tmp[col]<=ser_quantiles[0.50]][self.str_target].mean()
				flt_mean_target_75 = df_tmp[df_tmp[col]<=ser_quantiles[0.75]][self.str_target].mean()
			# if object
			if str_dtype == 'O':
				val_min, val_max, val_std, val_mean, val_median = np.nan, np.nan, np.nan, np.nan, np.nan
				val_range = np.nan
				val_mode, int_n_unique = ser_col.mode().iloc[0], ser_col.nunique()
				flt_prop_unique = int_n_unique / len(ser_col.dropna())
				flt_prop_negative = np.nan 
				flt_prop_min = np.nan
				flt_prop_max = np.nan
				flt_prop_zero = np.nan
				flt_mean_target_25 = np.nan 
				flt_mean_target_50 = np.nan
				flt_mean_target_75 = np.nan
			# if dtm
			if str_dtype == 'datetime64[ns]':
				val_min, val_max, val_mean, val_std, val_median = ser_col.min(), ser_col.max(), ser_col.mean(), np.nan, np.nan
				val_range = val_max - val_min
				val_mode, int_n_unique = ser_col.mode().iloc[0], ser_col.nunique()
				flt_prop_unique = int_n_unique / len(ser_col.dropna())
				flt_prop_negative = np.nan 
				flt_prop_min = len(ser_col[ser_col==val_min]) / len(ser_col.dropna())
				flt_prop_max = len(ser_col[ser_col==val_max]) / len(ser_col.dropna())
				flt_prop_zero = np.nan
				flt_mean_target_25 = np.nan 
				flt_mean_target_50 = np.nan
				flt_mean_target_75 = np.nan
			# create row
			dict_row = {
				'feature': col,
				'dtype': str_dtype,
				'propna': flt_prop_na,
				'min': val_min,
				'max': val_max,
				'range': val_range,
				'std': val_std,
				'mean': val_mean,
				'median': val_median,
				'mode': val_mode,
				'n_unique': int_n_unique,
				'prop_unique': flt_prop_unique,
				'prop_negative': flt_prop_negative,
				'prop_min': flt_prop_min,
				'prop_max': flt_prop_max,
				'prop_zero': flt_prop_zero,
				'mean_target_25': flt_mean_target_25,
				'mean_target_50': flt_mean_target_50,
				'mean_target_75': flt_mean_target_75,
			}
			# append
			list_dict_row.append(dict_row)
		# make df
		df_descriptives = pd.DataFrame(list_dict_row)
		# order cols
		df_descriptives.columns = [
			'feature',
			'dtype',
			'propna',
			'min',
			'max',
			'range',
			'std',
			'mean',
			'median',
			'mode',
			'n_unique',
			'prop_unique',
			'prop_negative',
			'prop_min',
			'prop_max',
			'prop_zero',
			'mean_target_25',
			'mean_target_50',
			'mean_target_75',
		]
		df_descriptives.sort_values(by='propna', ascending=False, inplace=True)
		df_descriptives.to_csv(f'{self.str_dirname_output}/{str_filename}', index=False)
		# save to object
		self.df_descriptives = df_descriptives
		# return object
		return self
	# match description
	def get_description(self, df, dict_match, str_feature='feature', str_filename='df_descriptives.csv'):
		# iterate through features
		list_str_description = []
		for str_feature in tqdm (df[str_feature]):
			# raw features
			if 'ENG-' not in str_feature:
				try:
					str_description = dict_match[str_feature]
				except KeyError:
					str_description = np.nan
			# inter-col FE (div)
			elif ('ENG-' in str_feature) and ('-div-' in str_feature):
				# split by -
				list_str_feature = str_feature.split('-')
				# numerator
				str_feature_numerator = list_str_feature[1]
				try:
					str_description_numerator = dict_match[str_feature_numerator]
				except KeyError:
					str_description_numerator = np.nan
				# denominator
				str_feature_denominator = list_str_feature[3]
				try:
					str_description_denominator = dict_match[str_feature_denominator]
				except KeyError:
					str_description_denominator = np.nan
				str_description = f'{str_description_numerator} divided by {str_description_denominator}'
			# engineered - cyclic
			elif ('ENG-' in str_feature) and (self.str_datecol in str_feature):
				str_description = 'Cyclic date feature'
			# else
			else:
				str_description = 'ERROR: CONDITION NOT ACCOUNTED FOR'
			# append to list
			list_str_description.append(str_description)
		# make col
		df['description'] = list_str_description
		df.to_csv(f'{self.str_dirname_output}/{str_filename}', index=False)
		# return object
		return self


	# df info summary table
	def get_df_info_summary_table(self, dict_df_info_train, dict_df_info_valid, dict_df_info_test):
		# create df
		df_info_summary = pd.DataFrame({
			'Data Set': ['Train','Valid','Test'],
			'Rows': [dict_df_info_train['int_nrows'], dict_df_info_valid['int_nrows'], dict_df_info_test['int_nrows']],
			'Columns': [dict_df_info_train['int_ncols'], dict_df_info_valid['int_ncols'], dict_df_info_test['int_ncols']],
			'Total Obs.': [dict_df_info_train['int_obs_total'], dict_df_info_valid['int_obs_total'], dict_df_info_test['int_obs_total']],
			'Min. Date': [dict_df_info_train['date_min'], dict_df_info_valid['date_min'], dict_df_info_test['date_min']],
			'Max. Date': [dict_df_info_train['date_max'], dict_df_info_valid['date_max'], dict_df_info_test['date_max']],
			'Prop. NaN': [dict_df_info_train['flt_propna'], dict_df_info_valid['flt_propna'], dict_df_info_test['flt_propna']],
			'Target Mean': [dict_df_info_train['flt_mean_target'], dict_df_info_valid['flt_mean_target'], dict_df_info_test['flt_mean_target']],
		})
		# save
		df_info_summary.to_csv(f'{self.str_dirname_output}/df_info_summary.csv', index=False)
		# return object
		return self


'''
to use previous class
1. call the current class
	cls_current = current_module(arguments=arguments)

2. call previous module from current class, assign to new object perhaps with previous module name
	cls_previous = cls_current.previous_module(argument=argument...)

3. make sure to review arguments from previous module in order to update:
	file name, path, etc. 


'''