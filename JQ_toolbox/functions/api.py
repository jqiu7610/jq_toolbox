# api
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from io import StringIO
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import catboost as cb
import ast
from itertools import chain
import time
pd.set_option('mode.chained_assignment', None)

# define generic transformer class
class GenericTransformer(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, list_transformers):
		self.list_transformers = list_transformers
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# loop through transformers
		time_start = time.perf_counter()
		for transformer in self.list_transformers:
			# transform
			X = transformer.transform(X)
		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		print(f'Generic Transformer: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		# return
		return X

# define function for payload parsing and generating output
class ParsePayload:
	# initialize
	def __init__(self, list_feats_raw_app, 
			           list_feats_raw_inc, 
					   list_feats_agg_inc,
					   dict_income_agg,
					   list_feats_raw_ln,
					   list_feats_raw_tuaccept,
					   list_feats_raw_cvlink,
					   df_empty,
					   pipeline_shared,
					   pipeline_pd,
					   pipeline_lgd,
					   list_non_numeric_pd,
					   list_string_cols,
					   dict_aa_pd,
					   dict_tiers,
					   bool_debt=False,
					   list_feats_raw_debt=None,
					   list_feats_agg_debt=None,
					   dict_debt_agg=None):
		# args
		self.list_feats_raw_app = list_feats_raw_app
		self.list_feats_raw_inc = list_feats_raw_inc
		self.list_feats_agg_inc = list_feats_agg_inc
		self.dict_income_agg = dict_income_agg
		self.list_feats_raw_debt = list_feats_raw_debt
		self.list_feats_agg_debt = list_feats_agg_debt
		self.dict_debt_agg = dict_debt_agg
		self.list_feats_raw_ln = list_feats_raw_ln
		self.list_feats_raw_tuaccept = list_feats_raw_tuaccept
		self.list_feats_raw_cvlink = list_feats_raw_cvlink
		self.df_empty = df_empty
		self.pipeline_shared = pipeline_shared
		self.pipeline_pd = pipeline_pd
		self.pipeline_lgd = pipeline_lgd
		self.list_string_cols = list_string_cols
		self.list_non_numeric_pd = list_non_numeric_pd
		self.dict_aa_pd = dict_aa_pd
		self.dict_tiers = dict_tiers
		self.bool_debt = bool_debt
	# payload for each applicant
	def get_payload_df(self, json_str_request):
		# get the payload for each applicant
		time_start = time.perf_counter()
		list_unique_id = []
		list_payload = []
		for applicant in json_str_request['rows']:
			# get unique_id
			unique_id = applicant['row_id']
			list_unique_id.append(unique_id)
			# get data tables
			payload = applicant['sources']
			list_payload.append(payload)
		# save output to self
		self.list_unique_id = list_unique_id
		self.list_payload = list_payload
		# time to get payloads
		flt_sec_get_payloads = time.perf_counter()-time_start
		self.flt_sec_get_payloads = flt_sec_get_payloads
		print(f'Time to get payloads: {flt_sec_get_payloads:0.5} sec.')
		# return object
		return self
	# define parse_application
	def parse_application(self, str_values):
		# create error
		self.error_app = ''
		# put into df
		df_app = pd.read_csv(StringIO(str_values), delimiter=',', usecols=lambda col: col.lower() in self.list_feats_raw_app)
		# convert to lower
		df_app.columns = df_app.columns.str.lower()
		# append __app to each column name except ApplicationDate
		df_app.columns = [f'{col}__app' for col in df_app.columns]
		# if df_app is all na append an error
		if df_app.isnull().all().all():
			self.error_app = 'No application data'
		# reset index
		df_app.reset_index(drop=True, inplace=True)
		# save df_app to self
		self.df_app = df_app
		# return object
		return self
	# define parse_income
	def parse_income(self, str_values):
		# create error
		self.error_inc = ''
		# put into df
		df_inc = pd.read_csv(StringIO(str_values), delimiter=',', usecols=lambda col: col.lower() in self.list_feats_raw_inc)
		# convert to lower
		df_inc.columns = df_inc.columns.str.lower()
		# check if df is empty
		if df_inc.empty:
			# create error
			self.error_inc = 'No income data'
			# create empty df with list_feats_inc_agg as cols
			df_inc = pd.DataFrame(columns=self.list_feats_agg_inc)
			# create row with nan
			df_inc = df_inc.append(pd.Series(), ignore_index=True)
		else:
			# filter rows
			df_inc = df_inc[(df_inc['bitinvalid']==False) & (df_inc['bituse']==True)]
			# if df_inc is empty after filtering on bit cols
			if df_inc.empty:
				# create error
				self.error_inc = 'No income data due to bit filter'
				# create empty df with list_feats_inc_agg as cols
				df_inc = pd.DataFrame(columns=self.list_feats_agg_inc)
				# create row with nan
				df_inc = df_inc.append(pd.Series(), ignore_index=True)
			else:
				# append __income to each column name
				df_inc.columns = [f'{col}__income' for col in df_inc.columns]
				# aggregate
				df_inc = df_inc.groupby('uniqueid__income', as_index=False).agg(self.dict_income_agg)
				# rename columns
				df_inc.columns = [f'{tpl_col[0]}_{tpl_col[1]}' for tpl_col in list(df_inc.columns)]
		# make sure bottom row is selected
		df_inc = pd.DataFrame(df_inc.iloc[-1, :]).T
		# reset index
		df_inc.reset_index(drop=True, inplace=True)
		# save to object
		self.df_inc = df_inc
		# return object
		return self	
	# define parse_debt
	def parse_debt(self, str_values):
		# create error
		self.error_debt = ''
		# put into df
		df_debt = pd.read_csv(StringIO(str_values), delimiter=',', usecols=lambda col: col.lower() in self.list_feats_raw_debt)
		# convert to lower
		df_debt.columns = df_debt.columns.str.lower()
		# check if df is empty
		if df_debt.empty:
			# create error
			self.error_debt = 'No debt data'
			# create empty df with list_feats_debt_agg as cols
			df_debt = pd.DataFrame(columns=self.list_feats_agg_debt)
			# create row with nan
			df_debt = df_debt.append(pd.Series(), ignore_index=True)
		else:
			# filter rows
			df_debt = df_debt[(df_debt['bitinvalid']==False) & (df_debt['bituse']==True)]
			# if df_debt is empty after filtering on bit cols
			if df_debt.empty:
				# create errors
				self.error_debt = 'No debt data due to bit filter'
				# create empty df with list_feats_debt_agg as cols
				df_debt = pd.DataFrame(columns=self.list_feats_agg_debt)
				# create row with nan
				df_debt = df_debt.append(pd.Series(), ignore_index=True)
			else:
				# append __debt to each column name
				df_debt.columns = [f'{col}__debt' for col in df_debt.columns]
				# aggregate
				df_debt = df_debt.groupby('uniqueid__debt', as_index=False).agg(self.dict_debt_agg)
				# rename columns
				df_debt.columns = [f'{tpl_col[0]}_{tpl_col[1]}' for tpl_col in list(df_debt.columns)]
		# make sure bottom row is selected
		df_debt = pd.DataFrame(df_debt.iloc[-1, :]).T
		# reset index
		df_debt.reset_index(drop=True, inplace=True)
		# save to object
		self.df_debt = df_debt
		# return object
		return self
	# define parse_ln
	def parse_ln(self, str_values):
		# create error
		self.error_ln = ''
		# put into df
		df_ln = pd.read_csv(StringIO(str_values), delimiter=',', usecols=lambda col: col.lower() in self.list_feats_raw_ln)
		# convert to lower
		df_ln.columns = df_ln.columns.str.lower()
		# append __ln to each column name
		df_ln.columns = [f'{col}__ln' for col in df_ln.columns]
		# check if df is empty
		if df_ln.empty:
			# create error
			self.error_ln = 'No Lexis Nexis data'
			# create row with nan
			df_ln = df_ln.append(pd.Series(), ignore_index=True)
		# make sure bottom row is selected
		df_ln = pd.DataFrame(df_ln.iloc[-1, :]).T
		# reset index
		df_ln.reset_index(drop=True, inplace=True)
		# save to object
		self.df_ln = df_ln
		# return object
		return self
	# define parse_tuxml
	def parse_tuxml(self, str_values):
		# create error
		self.error_tuxml = ''
		# define helper function for parsing XML
		def parse_xml(root, list_feats_raw_tuaccept, list_feats_raw_cvlink):
			# save child table vals so we don't have to write it a bunch
			str_child_table_vals = '{http://www.transunion.com/namespace}value'
			# get col names and values
			list_col_name = []
			list_col_value = []
			# iterate through child branches
			for child in root.iter(tag='{http://www.transunion.com/namespace}characteristic'): # child table
				# get col name
				col_name = child.find('{http://www.transunion.com/namespace}id').text # child table cols
				# if col in cv link
				if col_name.lower() in list_feats_raw_cvlink:
					# append __cvlink
					col_name = f'{col_name.lower()}__tucvlink'
					bool_in_payload = True
				# if col in tuaccept
				elif col_name.lower() in list_feats_raw_tuaccept:
					# append __tuaccept
					col_name = f'{col_name.lower()}__tuaccept'
					bool_in_payload = True
				# if not in either
				else:
					bool_in_payload = False
				# logic
				if bool_in_payload:
					# append col_name
					list_col_name.append(col_name)
					# try getting the value
					try:
						# get col value
						col_value = child.find(str_child_table_vals).text # child table vals
					# if its None
					except AttributeError:
						# set col value to np.nan
						col_value = np.nan
					# append col_value to list_col_value
					list_col_value.append(col_value)
			# define helper for helper function for converting to proper data type
			def convert_proper_dtype(str_value):
				# try converting to integer
				try:
					col_val = int(str_value)
				# if it cannot be converted to integer
				except:
					# try converting to float
					try:
						col_val = float(str_value)
					# if it cannot be converted to float
					except:
						col_val = str_value
				# return
				return col_val
			# convert each string value in list_col_value to appropriate value
			list_col_value = list(pd.Series(list_col_value).apply(lambda x: convert_proper_dtype(str_value=x)))
			# zip into dictionary
			dict_ = dict(zip(list_col_name, list_col_value))
			# put into df
			return pd.DataFrame(dict_, index=[0])
		# read data from string
		root = ET.fromstring(str_values)
		# parse xml
		df_tuxml = parse_xml(root=root, 
							 list_feats_raw_tuaccept=self.list_feats_raw_tuaccept,
							 list_feats_raw_cvlink=self.list_feats_raw_cvlink)
		# if df_tuxml is empty
		if df_tuxml.empty:
			# create error
			self.error_tu = 'No TU data'
			# create a row with nan
			df_tuxml = df_tuxml.append(pd.Series(), ignore_index=True)
		# make sure bottom row is selected
		df_tuxml = pd.DataFrame(df_tuxml.iloc[-1, :]).T
		# reset index
		df_tuxml.reset_index(drop=True, inplace=True)
		# save to object
		self.df_tuxml = df_tuxml
		# return object
		return self
	# define parse all
	def parse_all(self, json_str_request):
		# get payload df
		self.get_payload_df(json_str_request=json_str_request)
		# empty lists of lists
		time_start = time.perf_counter()
		list_list_errors = []
		list_list_df = []
		# iterate through payloads
		for payload in self.list_payload:
			# empty lists
			list_errors = []
			list_df = []
			# iterate through the tables
			for dict_data in payload:
				# get values
				str_values = dict_data['values']
				# application
				if dict_data['name'] == 'Application':
					self.parse_application(str_values=str_values)
					# append
					list_errors.append(self.error_app)
					list_df.append(self.df_app)
				# income
				if dict_data['name'] == 'Incomes':
					# parse income
					self.parse_income(str_values=str_values)
					# append 
					list_errors.append(self.error_inc)
					list_df.append(self.df_inc)
				# debt
				if (dict_data['name'] == 'Debts') and (self.bool_debt):
					# parse debt
					self.parse_debt(str_values=str_values)
					# append 
					list_errors.append(self.error_debt)
					list_df.append(self.df_debt)
				# ln
				if dict_data['name'] == 'Lexis Nexis Risk View 5':
					# parse ln
					self.parse_ln(str_values=str_values)
					# append
					list_errors.append(self.error_ln)
					list_df.append(self.df_ln)
				# tu
				if dict_data['name'] == 'TUXML':
					# parse xml
					self.parse_tuxml(str_values=str_values)
					# append
					list_errors.append(self.error_tuxml)
					list_df.append(self.df_tuxml)
			# append list_errors to list_list_errors
			list_list_errors.append(list_errors)
			# append list_df to list_list_df
			list_list_df.append(list_df)
		# save lists to object
		self.list_list_errors = list_list_errors
		self.list_list_df = list_list_df
		# time
		flt_sec_parse = time.perf_counter()-time_start
		self.flt_sec_parse = flt_sec_parse
		print(f'Time to parse data: {flt_sec_parse:0.5} sec.')
		# return object
		return self
	# define create_x
	def create_x(self, json_str_request):
		# parse all
		self.parse_all(json_str_request=json_str_request)
		# concatenate columns of each df in each list (horizontally i.e., axis=1)
		time_start = time.perf_counter()
		list_df_concat = []
		for list_dfs in self.list_list_df:
			# concatenate columns (i.e., column bind)
			df_concat = pd.concat(list_dfs, axis=1, sort=False) # WORKING PROPERLY		
			# append to list
			list_df_concat.append(df_concat)
		# concatenate rows (vertically)
		X = pd.concat(list_df_concat, axis=0, sort=False)
		# ensure there is a field for every feature
		X = pd.concat([self.df_empty, X], axis=0, sort=False) # WORKING PROPERLY
		# get list of feats missing
		ser_na = X.isnull().sum()
		# get those missing
		ser_na = ser_na[ser_na>0]
		# zip into dictionary
		dict_n_miss = dict(zip(ser_na.index, ser_na.values))
		# save to object
		self.X = X
		self.dict_n_miss = dict_n_miss
		# time
		flt_sec_create_x = time.perf_counter()-time_start
		self.flt_sec_create_x = flt_sec_create_x
		print(f'Time to create X: {flt_sec_create_x:0.5} sec.')
		# return object
		return self
	# define shared preprocessing
	def shared_preprocessing(self, json_str_request):
		# create X
		self.create_x(json_str_request=json_str_request)
		time_start = time.perf_counter()
		# transform
		X = self.pipeline_shared.transform(X=self.X)
		# lowercase cols names
		X.columns = [col.lower() for col in X.columns]

		# set string cols
		X[self.list_string_cols] = X[self.list_string_cols].astype(str)

		# save to object
		self.X = X
		# time
		flt_sec_preprocessing = time.perf_counter()-time_start
		self.flt_sec_preprocessing = flt_sec_preprocessing
		print(f'Time to complete shared preprocessing: {flt_sec_preprocessing:0.5} sec.')
		# return object
		return self
	# define counter_offers
	def counter_offers(self, json_str_request, int_n_samples=100):
		# shared preprocessing
		self.shared_preprocessing(json_str_request=json_str_request)
		time_start = time.perf_counter()
		# create large df
		X_lg = self.X.iloc[np.tile(np.arange(len(self.X)), int_n_samples)] # int_n_samples rows
		# make arry for keeping samples straight
		list_sample = list(np.repeat(list(range(1, (int_n_samples+1))), self.X.shape[0])) # start at 1 because we assign original to 0 below
		# set as sample
		X_lg['sample'] = list_sample

		# make array of loan to value
		list_ltv = list(np.linspace(0.5, 1.6, int_n_samples))
		# get n debtors
		int_n_debtors = self.X.shape[0]
		# duplicate each value
		list_ltv = np.repeat(list_ltv, int_n_debtors)
		# put into X
		X_lg['eng_loan_to_value'] = list_ltv

		# get amt financed (original)
		flt_amt_financed = X_lg['fltamountfinanced__app'].iloc[0]
		# get down total (original)
		flt_down_total = X_lg['fltapproveddowntotal__app'].iloc[0]

		# calculate amount financed
		X_lg['fltamountfinanced__app'] = X_lg['eng_loan_to_value'] * X_lg['fltapprovedpricewholesale__app']

		# calculate down pmt
		X_lg['fltapproveddowntotal__app'] = flt_amt_financed - X_lg['fltamountfinanced__app'] + flt_down_total

		# filter X_lg where fltapproveddowntotal__app > flt_down_total
		X_lg = X_lg[X_lg['fltapproveddowntotal__app'] > flt_down_total]

		# get list_transformers
		list_transformers = self.pipeline_shared.list_transformers

		# get binner
		cls_binner = list_transformers[5]
		# bin
		X_lg = cls_binner.transform(X_lg)

		# get val replacer
		cls_val_replacer = list_transformers[6]
		# replace in case things are binned to 0
		X_lg = cls_val_replacer.transform(X_lg)

		# get feature engineering class
		cls_feat_eng = list_transformers[7]
		# FE
		X_lg = cls_feat_eng.transform(X_lg)

		# subset to LTV bounds
		X_lg = X_lg[(X_lg['eng_loan_to_value']>0) & (X_lg['eng_loan_to_value']<=1.6)]

		# concatenate original X
		X_lg = pd.concat([self.X, X_lg])
		# fill na (sample)
		X_lg['sample'].fillna(0, inplace=True)

		# get pd model
		model_pd = self.pipeline_pd.model
		# predict
		X_lg['y_hat_pd'] = list(model_pd.predict_proba(X_lg[model_pd.feature_names_])[:,1])

		# get lgd model
		model_lgd = self.pipeline_lgd.model
		# predictions pd and clip to make sure no values <0 or >1
		X_lg['y_hat_lgd'] = list(np.clip(a=np.array(model_lgd.predict(X_lg[model_lgd.feature_names_])),
										 a_min=0,
										 a_max=1))
		# get original y_hat_pd (for all debtors)
		self.y_hat_pd = list(X_lg[X_lg['sample']==0]['y_hat_pd'])
		# get original y_hat_lgd (for all debtors)
		self.y_hat_lgd = list(X_lg[X_lg['sample']==0]['y_hat_lgd'])

		# group by sample so we cn get mean by account
		X_lg_grouped = X_lg[['fltapproveddowntotal__app',
							 'fltamountfinanced__app',
							 'fltapprovedpricewholesale__app',
							 'y_hat_pd',
							 'y_hat_lgd',
							 'sample']].groupby('sample', as_index=False).mean()

		# calculate ecnl
		X_lg_grouped['ecnl'] = X_lg_grouped['y_hat_pd'] * X_lg_grouped['y_hat_lgd']
		# calculate modified ecnl
		X_lg_grouped['ecnl_mod'] = (1.95553 * X_lg_grouped['ecnl']) - 0.03281

		# get ecnl
		self.y_hat_pd_x_lgd = list(X_lg_grouped[X_lg_grouped['sample']==0]['ecnl'])[0]
		# get modified cnl
		self.y_hat_pd_x_lgd_mod = list(X_lg_grouped[X_lg_grouped['sample']==0]['ecnl_mod'])[0]

		# save to object at this point for testing
		self.X_lg_grouped_pre_sub = X_lg_grouped.copy()

		# subset ecnl to < original
		X_lg_grouped = X_lg_grouped[(X_lg_grouped['ecnl'] < self.y_hat_pd_x_lgd) & (X_lg_grouped['fltapproveddowntotal__app'] >= flt_down_total)]

		# get date from row_id
		int_date = int(self.list_unique_id[0].split('_')[4])

		# get the keys in dict_tiers
		list_keys = list(self.dict_tiers.keys())

		# if theres only 1 key
		if len(list_keys) == 1:
			# get max of list_keys
			int_max_list_keys = np.max(list_keys)
		elif len(list_keys) > 1:
			# get keys <= int_date
			list_keys = [key for key in list_keys if key <= int_date]
			# get max of list_keys
			int_max_list_keys = np.max(list_keys)
		else:
			raise ValueError('Tier dictionary is empty')

		# get dict_tiers_subset from dict_tiers with key = int_max_list_keys
		dict_tiers_subset = self.dict_tiers[int_max_list_keys]

		# convert dict_tiers_subset to series and then sort by value ascending for iterating in the upcoming function
		ser_tiers = pd.Series(dict_tiers_subset).sort_values(ascending=True, inplace=False)

		# define function to map ecnl modified to tier
		def ecnl_to_tier(ser_tiers, flt_ecnl):
			# return declines right away
			if flt_ecnl > np.max(ser_tiers):
				return 'Decline'
			else:
				# iterate through values
				for idx, val in ser_tiers.items():
					# logic
					if flt_ecnl <= val:
						return idx

		# apply function
		X_lg_grouped['Tier'] = X_lg_grouped['ecnl_mod'].apply(lambda x: ecnl_to_tier(ser_tiers=ser_tiers, 
																		     		 flt_ecnl=x))

		# get indices of max ecnl_mod by tier
		ser_idx_max = X_lg_grouped.groupby('Tier')['ecnl_mod'].idxmax()
		# subset X_lg_grouped
		X_lg_grouped = X_lg_grouped.loc[ser_idx_max]

		# sort by ecnl_mod
		X_lg_grouped.sort_values(by='ecnl_mod', ascending=True, inplace=True)

		# save to object
		self.X_lg_grouped = X_lg_grouped
		# application date
		self.int_date = int_date
		# tiers used
		self.dict_tiers_subset = dict_tiers_subset
		# time
		flt_sec_counter = time.perf_counter()-time_start
		self.flt_sec_counter = flt_sec_counter
		print(f'Time to generate counter-offers: {flt_sec_counter:0.5} sec.')
		# return object
		return self
	# define adverse_action
	def adverse_action(self, json_str_request):
		# generate predictions
		self.counter_offers(json_str_request=json_str_request, int_n_samples=100)
		time_start = time.perf_counter()
		# get list of feats in model
		list_x_feats = self.pipeline_pd.model.feature_names_
		# pool X for readability
		X_pooled = cb.Pool(self.X[list_x_feats], cat_features=self.list_non_numeric_pd)
		# generate shap vals
		df_shap_vals = pd.DataFrame(self.pipeline_pd.model.get_feature_importance(data=X_pooled,
																			      type='ShapValues',
																			      prettified=False,
																				  thread_count=-1,
																			      verbose=False)).iloc[:, :-1]
		# set col names in df_shap_vals (setting to adverse actions could be faster)
		df_shap_vals.columns = list_x_feats
		# get reasons
		list_list_reasons = []
		for a, row in df_shap_vals.iterrows():
			# sort descending
			row_sorted = row.sort_values(ascending=False, inplace=False)
			# get top 5 features
			list_reasons = list(row_sorted[:5].index)
			# append to list_list_reasons
			list_list_reasons.append(list_reasons)
		# map features to reasons
		list_list_reasons = [list(pd.Series(list_reasons).map(self.dict_aa_pd)) for list_reasons in list_list_reasons]
		# save to object
		self.list_list_reasons = list_list_reasons
		# time
		flt_sec_adv_act = time.perf_counter()-time_start
		self.flt_sec_adv_act = flt_sec_adv_act
		print(f'Time to get adverse action: {flt_sec_adv_act:0.5} sec.')
		# return object
		return self
	# define generate output
	def generate_output(self, json_str_request):
		# get adverse action
		self.adverse_action(json_str_request=json_str_request)
		# start timer
		time_start = time.perf_counter()
		# create df
		df_output = pd.DataFrame({'Row_id': self.list_unique_id,
								  'Score_pd': self.y_hat_pd,
								  'Score_lgd': self.y_hat_lgd,
							      'Score_ecnl': [self.y_hat_pd_x_lgd for id_ in self.list_unique_id],
							      'Score_ecnl_mod': [self.y_hat_pd_x_lgd_mod for id_ in self.list_unique_id],
								  'Key_factors': self.list_list_reasons,
								  'Outlier_score': [0.0 for id_ in self.list_unique_id]})
		# convert to json
		str_output_ = df_output.to_json(orient='records')
		# convert to list
		list_output = ast.literal_eval(str_output_)
		# combine list of errors
		list_errors_final = list(chain(*self.list_list_errors))
		# remove empty strings from list_errors_final
		list_errors_final = [err for err in list_errors_final if err != '']
		# subset to cols we want
		X_lg_grouped_sub = self.X_lg_grouped[['fltapproveddowntotal__app',
										 	  'fltamountfinanced__app',
										 	  'fltapprovedpricewholesale__app',
										 	  'ecnl_mod',
										 	  'Tier']]
		# rename columns
		X_lg_grouped_sub.columns = ['Cash Down',
									'Amount Financed',
									'Price Wholesale',
									'ECNL Modified',
									'Tier']
		# create final output
		output_final = {"Request_id": "",
				        "Zaml_processing_id": "",
						"Response": [{"Model_name":"prestige-GenXI",
					                  "Model_version":"v2",
									  "Results":list_output,
									  "Errors":list_errors_final,
									  "Counter-Offers":X_lg_grouped_sub.to_dict('records')}]}
		# save to object
		self.output_final = output_final
		# time
		flt_sec_gen_output = time.perf_counter()-time_start
		self.flt_sec_gen_output = flt_sec_gen_output
		print(f'Time to generate output: {flt_sec_gen_output:0.5} sec.')
		# return object
		return self
