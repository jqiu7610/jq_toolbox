from .general_1 import General
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
import seaborn as sns

# define EDA class that will inherit from general class

class Exploratory_Data_Analysis(General):
	# init
	def __init__(self, str_dirname_output='./opt/ml/model', str_target='target', str_datecol='date', bool_target_binary=True):
		# use super to inithe parent class
		General.__init__(self, str_dirname_output)
		self.str_dirname_output = str_dirname_output
		self.str_target = str_target
		self.str_datecol = str_datecol
		self.bool_target_binary = bool_target_binary

	# get df info
	def get_df_info(self, df):
		# get rows and columns
		int_n_rows, int_n_columns = df.shape
		# get total objects
		int_objects_total = int_nrows * int_n_columns
		# get total missing
		int_n_missing = np.sum(df.isnull().sum())
		# min date
		min_date = np.min(df[self.str_datecol]),
		# max date
		max_date = np.max(df[self.str_datecol]),
		# mean target
		mean_target = np.mean(df[self.str_target])
		# % missing overall
		flt_propna = int_n_missing / int_objects_total
		# create dictionary
		dict_df_info = {
			'int_n_rows':int_n_rows,
			'int_n_columns':int_n_columns,
			'int_n_missing':int_n_missing,
			'min_date':min_date,
			'max_date':max_date,
			'mean_target':mean_target,
			'flt_propna':flt_propna,
		}

		# write to json
		json.dump(dict_df_info, open(f'{self.str_dirname_output}', 'w'))
		# return self
		return self

	# get proportion NaN by column
	def get_prop_nan_by_column(self, df, str_filename='df_propna.csv'):
		# create df
		df_propna = pd.DataFrame(df.isnull().mean())
		# cols
		df_propna.columns = ['prop_na']
		# sort
		df_propna.sort_avlues(by='prop_na', ascending=False, inplace=True)
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
		# close plot
		plt.close()
		# return object
		return self
	
	# plot target
	def plot_target(self, df, str_filename='plt_target.png'):
		# if we have a binary target
		if self.bool_target_binary:
			# get the total positive
			int_tot_pos = np.sum(df[self.str_target])
			# get total
			int_total = len(df[self.str_target])
			# get the toeal negative
			int_tot_neg = int_total - int_tot_pos
			# get pct negative class
			flt_pct_negative = (int_tot_neg / int_total) * 100
			# get pct positive class
			flt_pct_positive = (int_tot_pos / int_total) * 100
			# create axis
			fig, ax = plt.subplots(figsize=(15,10))
			# title
			ax.set_title(f'{flt_pct_negative:0.4}% = 0, {flt_pct_positive:0.4}% = 1, (N = {int_total})')
			# frequency bar plot
			ax.bar([0, 1], [int_tot_neg, int_tot_pos])
			# ylabel
			ax.set_ylabel('Frequency')
			# xticks
			ax.set_xticks([0, 1])
			# xtick labels
			ax.set_xticklabels(['0','1'])
		
		# if we have a continuous target
		else:
			# fig
			fig, ax = plt.subplots(figsize=(10,7))
			# title
			ax.set_title(f'Distribution of {self.str_target}')
			# plot
			sns.histplot(df[str_target], ax=ax, kde=True)
		# fix overlap
		fig.tight_layout()
		# save
		plt.savefig(f'{self.str_dirname_output}/{str_filename}', bbox_inches='tight')
		# close plot
		plt.close()
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