# feature engineering
import numpy as np
import time
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# class to codify strname__app into state codes
class Codify(BaseEstimator, TransformerMixin):
	# init
	def __init__(self, list_cols_codify='strname__app'):
		self.list_cols_codify = list_cols_codify

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		# start time
		time_start = time.perf_counter()

		X[self.list_cols_codify] = pd.Categorical(X[self.list_cols_codify])
		X['eng_state_code'] = X[self.list_cols_codify].cat.codes
		X.drop('strname__app', axis=1, inplace=True)
		
		# time stop
		time_stop = time.perf_counter()
		# timer
		timer = time_stop - time_start
		# print message
		print(f'Codifying state code: {timer:0.5} sec.')

		return X 

# class to codify unique string categorical features 
class DynamicEncode(BaseEstimator, TransformerMixin):
	# init
	def __init__(self, list_cols_codify):
		self.list_cols_codify = list_cols_codify

	# fit 
	def fit(self, X, y=None):
		return self

	# transform 
	def transform(self, X):
		# time start
		time_start = time.perf_counter()

		# loop through items in list
		for col in self.list_cols_codify.items():
			X[col] = pd.Categorical(X[col])
			X[f'eng_{col}_coded'] = X[col].cat.codes
			X.drop(col, axis=1, inplace=True)

		# time stop 
		time_stop = time.perf_counter()
		# timer 
		timer = time_stop - time_start
		# print message
		print(f'Codifying list of columns: {timer:0.5} sec')

		return X 


# create class for amortizing loans
class Amortization(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, int_month_max=72, str_col_amountfinanced='fltamountfinanced__app', str_col_payment='fltapprovedpayment__app', flt_apr=0.045, bool_ltv=True):
		self.int_month_max = int_month_max
		self.str_col_amtfinanced = str_col_amountfinanced
		self.str_col_payment = str_col_payment
		self.flt_apr = flt_apr
		self.bool_ltv = bool_ltv

	# fit
	def fit(self, X, y=None):
		return self

	# transform
	def transform(self, X):
		# time start
		time_start = time.perf_counter()

		# get montly rate
		flt_monthly_rate = self.flt_apr / 12

		# get monthly payment based on 72 months and apr
		X['adjusted_payment'] = (flt_monthly_rate * X[self.str_col_amtfinanced]) / (1 - (1 + flt_monthly_rate)**-self.int_month_max)

		# set unpaid principal at time 0
		X['eng_unpaid_principal_0'] = X[self.str_col_amtfinanced]

		# iterate through the months
		for month in range(1, self.int_month_max+1):

			# get monthly interest
			X['amount_to_interest'] = X[f'ENG-unpaid_principal_{int_month-1}'] * flt_monthly_rate
			
			# get principal reduction
			X['principal_reduction'] = X['adjusted_payment'] - X['amount_to_interest']

			# get unpaid principal
			X[f'eng_unpaid_principal_{month}'] = X[f'eng_unpaid_principal_{month-1}'] - X['principal_reduction']

			# get proportion unpaid principal
			X[f'eng_unpaid_principal_{int_month}'] = X[f'eng_unpaid_principal_{int_month}'] / X[self.str_col_amtfinanced]
			# if doing ltv
			if self.bool_ltv:
				try:
					# get loan (unpaid principal) to value at each month
					X[f'eng_ltv_value_{int_month}'] = X[f'eng_unpaid_principal_{int_month}'] / X[f' us used wholesale nat avg base monthly residuals as of value as-of date  - month {int_month}_newest']
				except KeyError:
					pass

		# drop feats
		X = X.drop(['adjusted_payment','eng_unpaid_principal_0', 'amount_to_interest', 'principal_reduction'], axis=1, inplace=False)

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		print(f'Amortization: {flt_sec:0.5} sec.')
		# return
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


# Target encoder
class TargetEncoder(BaseEstimator, TransformerMixin):
	def __init__(self, list_cols):
		self.list_cols = list_cols
		self.dict_col_all = {}
	def fit(self, X, y):
		# make col for y
		X['target'] = y
		# iterate
		for col in self.list_cols:
			# group
			X_grouped = X[[col,'target']].groupby(by=col).mean()
			# create dict
			dict_col = dict(zip(X_grouped.index, X_grouped['target']))
			# append to dict_col_all
			self.dict_col_all[col] = dict_col
		# drop target
		X.drop('target', axis=1, inplace=True)
		# return
		return self

	# transform
	def transform(self, X):
		# start timer
		time_start = time.perf_counter()

		# future proof
		list_cols = [col for col in self.list_cols if col in list(X.columns)]
		# replace
		X[list_cols] = X[list_cols].apply(lambda col: col.map(self.dict_col_all[col.name]))

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		print(f'Target Encoder: {flt_sec:0.5} sec.')
		# return
		return X


# creating theoretical LTV
class TheoreticalLTV(BaseEstimator, TransformerMixin):
	# init
	def __init__(self, int_months=72):
		self.int_months = int_months

	# fit
	def fit(self, X, y=None):
		return self

	#transform
	def transform(self, X):
		# start timer
		time_start = time.perf_counter()

		# loop
		for month in range(1, self.int_months+1):
			try:
				X[f'eng-ltv-monthlyresidual{month}__bb'] = X['fltamountfinanced__app'] / X[f'monthlyresidual{month}__bb']
			except:
				pass
		
		# timer end
		time_end = time.perf_counter()
		flt_sec = time_end - time_start

		# print message
		print(f'theoretical LTV: {flt_sec:0.5} sec.')

		return X

# creating theoretical LTV using unpaid principal
class TheoreticalLTV_new(BaseEstimator, TransformerMixin):
	# init
	def __init__(self, int_months=72):
		self.int_months = int_months

	# fit
	def fit(self, X, y=None):
		return self

	# transform
	def transform(self, X):
		# start timer
		time_start = time.perf_counter()

		#loop
		for month in range(1, self.int_months+1):
			try:
				X[f'eng-ltvnew-monthlyresidual{month}__bb'] = X[f'fltamountfinanced__app_'] / X[f'monthlyresidual{month}__bb']
			except:
				pass

		#timer end
		time_end = time.perf_counter()
		flt_sec = time_end - time_start

		#print message
		print(f'new LTV: {flt_sec:0.5} sec.')

		return X


class FeatureEngineeringJQ:
	# transform
	def transform(self, X):
		try:
			X['eng-PTI'] = X['fltapprovedpayment__app'] / X['fltgrossmonthly__income_sum']
		except:
			pass
		#LTV on residuals
		try:
			X['fltamountfinanced__app_new'] = X['fltamountfinanced__app'] - X['fltgapinsurance__app'] - X['fltservicecontract__app']
		except:
			pass
		
		# #TRADE INFO
		# # number of open trades / number of trades
		# try:
		# 	X['eng_at02s_to_at01s'] = X['at02s__tuaccept'] / X['at01s__tuaccept']
		# except:
		# 	pass
		# #satisfactory open trades / number of trade
		# try:
 	# 		X['eng_at03s_to_at01s'] = X['at03s__tuaccept'] / X['at01s__tuaccept']
		# except:
		# 	pass
		# #24 months open trade / number of trades
		# try:
 	# 		X['eng_at09s_to_at01s'] = X['at09s__tuaccept'] / X['at01s__tuaccept']
		# except:
		# 	pass
		# #open satisfactory 24 months / number of trades
		# try:
 	# 		X['eng_at27s_to_at01s'] = X['at27s__tuaccept'] / X['at01s__tuaccept']
		# except:
		# 	pass
		# #total past due amount of open trades / total balance of all trades in 12 months
		# try:	
		# 	X['eng_at57s_to_at01s'] = X['at57s__tuaccept'] / X['at101s__tuaccept']
		# except:
		# 	pass
		
		# #AUTO TRADE
		# # open auto vs number of auto trades
		# try:
		# 	X['eng_au02s_to_au01s'] = X['au02s__tuaccept'] / X['au01s__tuaccept']
		# except:
		# 	pass
		# # satisfactory auto trades over number auto trades
		# try:
		# 	X['eng_au03s_to_au01s'] = X['au03s__tuaccept'] / X['au01s__tuaccept']
		# except:
		# 	pass
		# # auto trades opened in 24 / number of auto trades
		# try:
		# 	X['eng_au09s_to_au01s'] = X['au09s__tuaccept'] / X['au01s__tuaccept']
		# except:
		# 	pass
		# # months since recent / months since oldest auto trade opened
		# try:
		# 	X['eng_au21s_to_au20s'] = X['au21s__tuaccept'] / X['au20s__tuaccept']
		# except:
		# 	pass
		# # open and satisf auto trades 24 months / number of auto trades
		# try:
		# 	X['eng_au27s_to_au01s'] = X['au27s__tuaccept'] / X['au01s__tuaccept']
		# except:
		# 	pass
		# # open and satisf auto trades 24 months / number of open auto trades
		# try:
		# 	X['eng_au27s_to_au02s'] = X['au27s__tuaccept'] / X['au02s__tuaccept']
		# except:
		# 	pass
		# # open and satisf auto trades 24 months / number of open satisf trades
		# try:
		# 	X['eng_au27s_to_au03s'] = X['au27s__tuaccept'] / X['au03s__tuaccept']
		# except:
		# 	pass
		
		# #CREDIT CARD TRADES
		# # open CC trades vs CC trades
		# try:
		# 	X['eng_bc02s_to_bc01s'] = X['bc02s__tuaccept'] / X['bc01s__tuaccept']
		# except:
		# 	pass
		# # current open satisf CC trades vs CC trades
		# try:
		# 	X['eng_bc03s_to_bc01s'] = X['bc03s__tuaccept'] / X['bc01s__tuaccept']
		# except:
		# 	pass
		# # open CC trades 24m vs CC trades
		# try:
		# 	X['eng_bc09s_to_bc01s'] = X['bc09s__tuaccept'] / X['bc01s__tuaccept']
		# except:
		# 	pass
		# # months since most recent vs month since oldest CC 
		# try:	
		# 	X['eng_bc21s_to_bc20s'] = X['bc21s__tuaccept'] / X['bc20s__tuaccept']
		# except:
		# 	pass
		# # open satisf CC trade 24 months vs CC trades
		# try:	
		# 	X['eng_bc27s_to_bc01s'] = X['bc27s__tuaccept'] / X['bc01s__tuaccept']
		# except:
		# 	pass
		
		# #BANK INSTALLMENTS
		# # open bank installment vs bank installment trades
		# try:	
		# 	X['eng_bi02s_to_bi01s'] = X['bi02s__tuaccept'] / X['bi01s__tuaccept']
		# except:
		# 	pass
		# # open satisf bank installment vs number bank installment
		# try:	
		# 	X['eng_bi12s_to_bi01s'] = X['bi12s__tuaccept'] / X['bi01s__tuaccept']
		# except:
		# 	pass
		# # months since most recent bi vs months since oldest bi
		# try:	
		# 	X['eng_bi21s_to_bi20s'] = X['bi21s__tuaccept'] / X['bi20s__tuaccept']
		# except:
		# 	pass
		# # utilization open bi verified 12m vs total open bi verified 12m
		# try:
		# 	X['eng_bi34s_to_bi33s'] = X['bi34s__tuaccept'] / X['bi33s__tuaccept']
		# except:
		# 	pass
		
		# #BANK REVOLVER
		# # open br trades vs br trades
		# try:	
		# 	X['eng_br02s_to_br01s'] = X['br02s__tuaccept'] / X['br01s__tuaccept']
		# except:
		# 	pass
		# # open satisf br trades vs br trades
		# try:
		# 	X['eng_br03s_to_br01s'] = X['br03s__tuaccept'] / X['br01s__tuaccept']
		# except:
		# 	pass
		# # br trades opened past 24 / br trades opened
		# try:
		# 	X['eng_br09s_to_br01s'] = X['br09s__tuaccept'] / X['br01s__tuaccept']
		# except:
		# 	pass
		# # months recent br / months oldest br
		# try:
		# 	X['eng_br21s_to_br20s'] = X['br21s__tuaccept'] / X['br20s__tuaccept']
		# except:
		# 	pass
		# # open satis 24m / br trades
		# try:
		# 	X['eng_br27s_to_br20s'] = X['br27s__tuaccept'] / X['br20s__tuaccept']
		# except:
		# 	pass
		
		# #CHARGE OFF TRADES
		# # number CO trades in past 24 months / CO trades
		# try:
		# 	X['eng_co03s_to_co01s'] = X['co03s__tuaccept'] / X['co01s__tuaccept']
		# except:
		# 	pass
		# # balance CO 24m / CO balance
		# try:
		# 	X['eng_co07s_to_co05s'] = X['co07s__tuaccept'] / X['co05s__tuaccept']
		# except:
		# 	pass
		
		# # FORECLOSURE TRADES
		# # foreclosure trades past 24m / foreclosure trades
		# try:
		# 	X['eng_fc03s_to_fc01s'] = X['fc03s__tuaccept'] / X['fc01s__tuaccept']
		# except:
		# 	pass
		# # balance FC trades 24m / balance FC trades
		# try:
		# 	X['eng_fc07s_to_fc05s'] = X['fc07s__tuaccept'] / X['fc05s__tuaccept']
		# except:
		# 	pass
		
		# #FINANCE INSTALLMENT
		# # open fi trades / fi trades
		# try:
		# 	X['eng_fi02s_to_fi01s'] = X['fi02s__tuaccept'] / X['fi01s__tuaccept']
		# except:
		# 	pass
		# # open satisf fi trades / fi trades
		# try:
		# 	X['eng_fi03s_to_fi01s'] = X['fi03s__tuaccept'] / X['fi01s__tuaccept']
		# except:
		# 	pass
		# # number fi opened in past 24 months / opened fi trades
		# try:	
		# 	X['eng_fi09s_to_fi02s'] = X['fi09s__tuaccept'] / X['fi02s__tuaccept']
		# except:
		# 	pass
		# # number fi opened past 24m / number fi trades
		# try:
		# 	X['eng_fi09s_to_fi01s'] = X['fi09s__tuaccept'] / X['fi01s__tuaccept']
		# except:
		# 	pass
		# # months most recent fi opened / months since oldest fi opened
		# try:
		# 	X['eng_fi21s_to_fi20s'] = X['fi21s__tuaccept'] / X['fi20s__tuaccept']
		# except:
		# 	pass
		# # number current open satisf fi 24m / number fi trades
		# try:
		# 	X['eng_fi27s_to_fi01s'] = X['fi27s__tuaccept'] / X['fi01s__tuaccept']
		# except:
		# 	pass
		
		# #FINANCE REVOLVING TRADES
		# # number of open FR trades / number FR trades
		# try:
		# 	X['eng_fr02s_to_fr01s'] = X['fr02s__tuaccept'] / X['fr01s__tuaccept']
		# except:
		# 	pass
		# # number current satisf open fr / number FR trades
		# try:
		# 	X['eng_fr03s_to_fr01s'] = X['fr03s__tuaccept'] / X['fr01s__tuaccept']
		# except:
		# 	pass
		# # number opened fr trades 24m / number FR trades
		# try:
		# 	X['eng_fr09s_to_fr01s'] = X['fr09s__tuaccept'] / X['fr01s__tuaccept']
		# except:
		# 	pass
		
		# # HOME EQUITY
		# # open home equity vs number of home equity loans
		# try:
		# 	X['eng_hi02s_to_hi01s'] = X['hi02s__tuaccept'] / X['hi01s__tuaccept']
		# except:
		# 	pass
		# # current satisf open he vs number he loans
		# try:
		# 	X['eng_hi03s_to_hi01s'] = X['hi03s__tuaccept'] / X['hi01s__tuaccept']
		# except:
		# 	pass
		# # number he opened past 24m / number he loans
		# try:
		# 	X['eng_hi09s_to_hi01s'] = X['hi09s__tuaccept'] / X['hi01s__tuaccept']
		# except:
		# 	pass
		# # months since most recent he opened / months since oldest
		# try:
		# 	X['eng_hi21s_to_hi20s'] = X['hi21s__tuaccept'] / X['hi20s__tuaccept']
		# except:
		# 	pass
		# # number currently open satisf he loan 24m / number he loans
		# try:
		# 	X['eng_hi27s_to_hi01s'] = X['hi27s__tuaccept'] / X['hi01s__tuaccept']
		# except:
		# 	pass
		
		# # HOME EQUITY LOC
		# # number he open LOC / number he LOC
		# try:
		# 	X['eng_hr02s_to_hr01s'] = X['hr02s__tuaccept'] / X['hr01s__tuaccept']
		# except:
		# 	pass
		# # number he opened LOC 24m / number he LOC
		# try:
		# 	X['eng_hr12s_to_hr01s'] = X['hr12s__tuaccept'] / X['hr01s__tuaccept']
		# except:
		# 	pass
		# # months since most recent opened vs months oldest
		# try:
		# 	X['eng_hr21s_to_hr20s'] = X['hr21s__tuaccept'] / X['hr20s__tuaccept']
		# except:
		# 	pass
		
		# # INSTALLMENT TRADES
		# # number open installments vs installment trades
		# try:
		# 	X['eng_in02s_to_in01s'] = X['in02s__tuaccept'] / X['in01s__tuaccept']
		# except:
		# 	pass
		# # current open satisf vs installment trades
		# try:
		# 	X['eng_in03s_to_in01s'] = X['in03s__tuaccept'] / X['in01s__tuaccept']
		# except:
		# 	pass
		# # number opened past 24m vs installment trades
		# try:
		# 	X['eng_in09s_to_in01s'] = X['in09s__tuaccept'] / X['in01s__tuaccept']
		# except:
		# 	pass
		# # number open verified in past 12 months vs installment trades
		# try:
		# 	X['eng_in12s_to_in01s'] = X['in12s__tuaccept'] / X['in01s__tuaccept']
		# except:
		# 	pass
		# # months since most recent vs months oldest
		# try:
		# 	X['eng_in21s_to_in20s'] = X['in21s__tuaccept'] / X['in20s__tuaccept']
		# except:
		# 	pass
		# # open satisf 24m vs installment trades
		# try:
		# 	X['eng_in27s_to_in01s'] = X['in27s__tuaccept'] / X['in01s__tuaccept']
		# except:
		# 	pass
		# # open verified 12m vs installment trades
		# try:
		# 	X['eng_in28s_to_in01s'] = X['in28s__tuaccept'] / X['in01s__tuaccept']
		# except:
		# 	pass
		
		# # LOAN MODIFICATIONS
		# # number LM mortage 90+DPD vs LM mortgage
		# try:
		# 	X['eng_lm08s_to_lm01s'] = X['lm08s__tuaccept'] / X['lm01s__tuaccept']
		# except:
		# 	pass
		# # bank backed LM vs LM 
		# try:
		# 	X['eng_lm25s_to_lm01s'] = X['lm25s__tuaccept'] / X['lm01s__tuaccept']
		# except:
		# 	pass
		
		# # MORTGAGE TRADES
		# # number of open mortgage trades vs number or mortgage trades
		# try:
		# 	X['eng_mt02s_to_mt01s'] = X['mt02s__tuaccept'] / X['mt01s__tuaccept']
		# except:
		# 	pass
		# #number of current satisf MT vs mortgage trades
		# try:
		# 	X['eng_mt03s_to_mt01s'] = X['mt03s__tuaccept'] / X['mt01s__tuaccept']
		# except:
		# 	pass
		# # mt trades opened in 24 months vs mt trades
		# try:
		# 	X['eng_mt09s_to_mt01s'] = X['mt09s__tuaccept'] / X['mt01s__tuaccept']
		# except:
		# 	pass
		# # open verified in past 12 months vs mortgage trades
		# try:
		# 	X['eng_mt12s_to_mt01s'] = X['mt12s__tuaccept'] / X['mt01s__tuaccept']
		# except:
		# 	pass
		# #months most recent opened vs oldest opened
		# try:
		# 	X['eng_mt21s_to_mt20s'] = X['mt21s__tuaccept'] / X['mt20s__tuaccept']
		# except:
		# 	pass
		# # number open satisf MT 24 months vs MT
		# try:
		# 	X['eng_mt27s_to_mt01s'] = X['mt27s__tuaccept'] / X['mt01s__tuaccept']
		# except:
		# 	pass
		
		# # joint trade info
		# # trades to joint trades open/satisf
		# try:
		# 	X['eng_at03s_to_jt03s'] = X['at03s__tuaccept'] / X['jt03s__tuaccept']
		# except:
		# 	pass 
		# # auto trades to joint trades open/satisf
		# try:
		# 	X['eng_au03s_to_jt03s'] = X['au03s__tuaccept'] / X['jt03s__tuaccept']
		# except:
		# 	pass
		# # credit card to joint trades open/satisf
		# try:
		# 	X['eng_bc03s_to_jt03s'] = X['bc03s__tuaccept'] / X['jt03s__tuaccept']
		# except:
		# 	pass
		# # bank installments vs JT open/satisf
		# try:
		# 	X['eng_bi03s_to_jt03s'] = X['bi03s__tuaccept'] / X['jt03s__tuaccept']
		# except:
		# 	pass 
		# # bank revolver vs JT open/satisf
		# try:
		# 	X['eng_br03s_to_jt03s'] = X['br03s__tuaccept'] / X['jt03s__tuaccept']
		# except:
		# 	pass
		# # charge offs in 24 months vs JT open/satisf
		# try:
		# 	X['eng_co03s_to_jt03s'] = X['co03s__tuaccept'] / X['jt03s__tuaccept']
		# except:
		# 	pass
		# # foreclosure in 24 months vs JT open/satisf
		# try:
		# 	X['eng_fc03s_to_jt03s'] = X['fc03s__tuaccept'] / X['jt03s__tuaccept']
		# except:
		# 	pass
		# # finance installments vs JT open/satisf
		# try:
		# 	X['eng_fi03s_to_jt03s'] = X['fi03s__tuaccept'] / X['jt03s__tuaccept']
		# except:
		# 	pass
		# # home equity v JT open/satisf
		# try:
		# 	X['eng_hi03s_to_jt03s'] = X['hi03s__tuaccept'] / X['jt03s__tuaccept']
		# except:
		# 	pass
		# # home equity LOC v JT open/satisf
		# try:
		# 	X['eng_hr03s_to_jt03s'] = X['hr03s__tuaccept'] / X['jt03s__tuaccept']
		# except:
		# 	pass
		# # installment trades v JT open/satisf
		# try:
		# 	X['eng_in03s_to_jt03s'] = X['in03s__tuaccept'] / X['jt03s__tuaccept']
		# except:
		# 	pass
		# # mortgage trades vs JT open/satisf
		# try:
		# 	X['eng_mt03s_to_jt03s'] = X['mt03s__tuaccept'] / X['jt03s__tuaccept']
		# except:
		# 	pass
		

		# #======================== Bankruptcy fields =========================================
		# #bankruptcy 24 months vs bankruptcy
		# try:
		# 	X['eng_g099s_to_g094s'] = X['g099s__tuaccept'] / X['g094s__tuaccept']
		# except:
		# 	pass
		# #number of trade bankruptcies vs number of trades
		# try:
		# 	X['eng_g100s_to_at01s'] = X['g100s__tuaccept'] / X['at01s__tuaccept']
		# except:
		# 	pass
		# #number of trade bankruptcies vs number of open trades
		# try:
		# 	X['eng_g100s_to_at02s'] = X['g100s__tuaccept'] / X['at02s__tuaccept']
		# except:
		# 	pass
		# #number of trade bankruptcies vs number of open/satisf trades
		# try:
		# 	X['eng_g100s_to_at03s'] = X['g100s__tuaccept'] / X['at03s__tuaccept']
		# except:
		# 	pass
		# #number of trade bankruptcies verified in past 24 months vs trades opened in past 24
		# try:
		# 	X['eng_g099a_to_at09s'] = X['g099a__tuaccept'] / X['at09s__tuaccept']
		# except:
		# 	pass
		# #number of trade bankruptcies verified in past 24 months vs open satisf trades 24
		# try:
		# 	X['eng_g099a_to_at27s'] = X['g099a__tuaccept'] / X['at27s__tuaccept']
		# except:
		# 	pass

		# # BK fields
		# # --DONE--X['g094s__tuaccept'] number of public record bankruptcies
		# # --DONE--X['g099s__tuaccept'] number of public BK past 24 months
		# # --DONE--X['g100s__tuaccept'] number of tradeline BK
		# # try:
		# # 	X['eng_inttype_times_LINKF032'] = X['intType__app'] * X['LINKF032__tucvlink']
		# # except:
		# # 	pass
		# # # months since most recent inquiry over months since most recent public record BK
		# try:
		# 	X['eng_g102s_to_s207s'] = X['g102s__tuaccept'] / X['s207s__tuaccept']
		# except:
		# 	pass
		# # months since most recent inquiry over months since most recent tradeline BK
		# try:
		# 	X['eng_g102s_to_s207a'] = X['g102s__tuaccept'] / X['s207a__tuaccept']
		# except:
		# 	pass
		# # =====================BK between CREDIT VISION VS FACTOR TRUST======================
		# # public record BK 24 months
		# try:
		# 	X['eng_linkt008_times_g099s'] = X['LINKT008__tucvlink'] * X['g099s__tuaccept']
		# except:
		# 	pass
		# # trade line BK
		# try:
		# 	X['eng_linkt009_times_g100s'] = X['LINKT009__tucvlink'] * X['g100s__tuaccept']
		# except:
		# 	pass
		# # public record BK	
		# try:
		# 	X['eng_linkt010_times_g094s'] = X['LINKT010__tucvlink'] * X['g094s__tuaccept']
		# except:
		# 	pass
		return X 
	

list_cols_eng_raw = ['fltAmountFinanced__app', 'fltApprovedPriceWholesale__app', 'jt03s__tuaccept',
					 'fltGrossMonthly__income_sum', 'fltApprovedDownTotal__app', 'fltAmountFinanced__app', 'fltApprovedDownTotal__app', 
					 'fltGrossMonthly__income_sum', 'fltGrossMonthly__income_std', 'fltGrossMonthly__income_min', 'fltGrossMonthly__income_median',
					 'fltGrossMonthly__income_mean', 'fltGrossMonthly__income_max', 'fltApprovedDownTotal__app', 'fltApprovedPriceWholesale__app', 'at02s__tuaccept', 
					 'at01s__tuaccept', 'at03s__tuaccept', 'at01s__tuaccept', 'at09s__tuaccept', 'at27s__tuaccept', 'at32s__tuaccept', 
					 'at57s__tuaccept', 'at33a__tuaccept', 'at33b__tuaccept', 'at34a__tuaccept', 'at34b__tuaccept', 'at35a__tuaccept',
					 'at35b__tuaccept', 'at101s__tuaccept', 'au02s__tuaccept', 'au01s__tuaccept', 'au03s__tuaccept', 'au34s__tuaccept', 
					 'au35s__tuaccept', 'au09s__tuaccept', 'au21s__tuaccept', 'au20s__tuaccept', 'au27s__tuaccept', 'au32s__tuaccept',
					 'au33s__tuaccept', 'bc02s__tuaccept', 'bc01s__tuaccept', 'bi03s__tuaccept', 'bc03s__tuaccept', 'bc09s__tuaccept', 
					 'bc21s__tuaccept', 'bc20s__tuaccept', 'bc27s__tuaccept', 'bc32s__tuaccept', 'bc33s__tuaccept', 'bc34s__tuaccept', 
					 'bc35s__tuaccept', 'bi02s__tuaccept', 'bi01s__tuaccept', 'bi12s__tuaccept', 'bi21s__tuaccept', 'bi20s__tuaccept', 
					 'bi32s__tuaccept', 'bi34s__tuaccept', 'bi33s__tuaccept', 'bi35s__tuaccept', 'br02s__tuaccept', 'br01s__tuaccept', 
					 'br03s__tuaccept', 'br09s__tuaccept', 'br21s__tuaccept', 'br20s__tuaccept', 'br27s__tuaccept', 'br32s__tuaccept', 
					 'br33s__tuaccept', 'br34s__tuaccept', 'br35s__tuaccept', 'co03s__tuaccept', 'co01s__tuaccept', 'co07s__tuaccept', 
					 'co05s__tuaccept', 'fc03s__tuaccept', 'fc01s__tuaccept', 'fc07s__tuaccept', 'fc05s__tuaccept', 'fi02s__tuaccept', 
					 'fi01s__tuaccept', 'fi03s__tuaccept', 'fi01s__tuaccept', 'fi09s__tuaccept', 'fi02s__tuaccept', 'fi09s__tuaccept', 
					 'fi21s__tuaccept', 'fi20s__tuaccept', 'fi27s__tuaccept', 'fi01s__tuaccept', 'fr02s__tuaccept', 'fr01s__tuaccept', 
					 'fr03s__tuaccept', 'fr09s__tuaccept', 'fr32s__tuaccept', 'fr33s__tuaccept', 'fr34s__tuaccept', 'fr35s__tuaccept', 
					 'fi32s__tuaccept', 'fi33s__tuaccept','fi34s__tuaccept', 'fi35s__tuaccept', 'hi02s__tuaccept', 'hi01s__tuaccept', 
					 'hi03s__tuaccept', 'hi01s__tuaccept', 'hi09s__tuaccept', 'hi21s__tuaccept', 'hi20s__tuaccept', 'hi27s__tuaccept', 
					 'hi32s__tuaccept', 'hi33s__tuaccept','hi34s__tuaccept', 'hi35s__tuaccept','hr02s__tuaccept', 'hr01s__tuaccept', 
					 'hr03s__tuaccept', 'hr12s__tuaccept', 'hr01s__tuaccept', 'hr21s__tuaccept', 'hr20s__tuaccept', 'hr32s__tuaccept', 
					 'hr34s__tuaccept', 'in02s__tuaccept', 'in01s__tuaccept', 'in03s__tuaccept', 'g100s__tuaccept', 'g099a__tuaccept',
					 'in09s__tuaccept', 'in12s__tuaccept', 'in21s__tuaccept', 'in20s__tuaccept', 'in27s__tuaccept', 'in28s__tuaccept', 
					 'in32s__tuaccept', 'in33s__tuaccept', 'in34s__tuaccept', 'in35s__tuaccept', 'lm08s__tuaccept', 'lm01s__tuaccept', 
					 'lm25s__tuaccept', 'mt02s__tuaccept', 'mt01s__tuaccept', 'mt03s__tuaccept', 'mt09s__tuaccept', 'mt12s__tuaccept', 
					 'mt21s__tuaccept', 'mt20s__tuaccept', 'mt27s__tuaccept', 'mt01s__tuaccept', 'mt32s__tuaccept', 'mt33s__tuaccept', 
					 'mt34s__tuaccept', 'mt35s__tuaccept', 'g099s__tuaccept', 'intType__app', 'LINKF032__tucvlink__bin', 'LINKT008__tucvlink',
					 'LINKT009__tucvlink', 'LINKT010__tucvlink', 'g094s__tuaccept', 'bankruptcycount24month__ln', 'bankruptcycount__ln',
					 'derogcount12month__ln', 'derogcount__ln', 'sourcecredheadertimenewest__ln', 'sourcecredheadertimeoldest__ln',
					 'sourcenonderogcount12month__ln', 'sourcenonderogcount__ln', 'assetproppurchasetimenewest__ln', 'assetproppurchasetimeoldest__ln',
					 'assetpropsaletimenewest__ln', 'assetpropsaletimeoldest__ln', 'lienjudgmentcount12month__ln', 'lienjudgmentcount__ln',
					 'shorttermloanrequest24month__ln', 'shorttermloanrequest__ln','evictioncount12month__ln', 'evictioncount__ln']

