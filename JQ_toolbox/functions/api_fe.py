# feature engineering
import numpy as np
import time
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineeringJQ:
	# transform
	def transform(self, X):
		# LTV
		try:
			X['eng_loan_to_value'] = X['fltamountfinanced__app'] / X['fltapprovedpricewholesale__app']
		except:
			pass
		# DTI - Not used, but here just in case
		# try:
		# 	X['eng_debt_to_income'] = X['fltMonthlyPayment__debt_mean'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# down payment to financed
		try:
			X['eng_down_to_financed'] = X['fltapproveddowntotal__app'] / X['fltamountfinanced__app']
		except:
			pass
		# down pmt over grossmonthly
		try:
			X['eng_down_to_income'] = X['fltapproveddowntotal__app'] / X['fltgrossmonthly__income_sum']
		except:
			pass
		# down pmt over price wholesale
		try:
			X['eng_down_to_wholesale'] = X['fltapproveddowntotal__app'] / X['fltapprovedpricewholesale__app']
		except:
			pass

		#TRADE INFO
		# number of open trades / number of trades
		try:
			X['eng_at02s_to_at01s'] = X['at02s__tuaccept'] / X['at01s__tuaccept']
		except:
			pass
		#satisfactory open trades / number of trade
		try:
 			X['eng_at03s_to_at01s'] = X['at03s__tuaccept'] / X['at01s__tuaccept']
		except:
			pass
		#24 months open trade / number of trades
		try:
 			X['eng_at09s_to_at01s'] = X['at09s__tuaccept'] / X['at01s__tuaccept']
		except:
			pass
		#open satisfactory 24 months / number of trades
		try:
 			X['eng_at27s_to_at01s'] = X['at27s__tuaccept'] / X['at01s__tuaccept']
		except:
			pass
		#total past due amount of open trades / total balance of all trades in 12 months
		try:	
			X['eng_at57s_to_at01s'] = X['at57s__tuaccept'] / X['at101s__tuaccept']
		except:
			pass
		
		#AUTO TRADE
		# open auto vs number of auto trades
		try:
			X['eng_au02s_to_au01s'] = X['au02s__tuaccept'] / X['au01s__tuaccept']
		except:
			pass
		# satisfactory auto trades over number auto trades
		try:
			X['eng_au03s_to_au01s'] = X['au03s__tuaccept'] / X['au01s__tuaccept']
		except:
			pass
		# auto trades opened in 24 / number of auto trades
		try:
			X['eng_au09s_to_au01s'] = X['au09s__tuaccept'] / X['au01s__tuaccept']
		except:
			pass
		# months since recent / months since oldest auto trade opened
		try:
			X['eng_au21s_to_au20s'] = X['au21s__tuaccept'] / X['au20s__tuaccept']
		except:
			pass
		# open and satisf auto trades 24 months / number of auto trades
		try:
			X['eng_au27s_to_au01s'] = X['au27s__tuaccept'] / X['au01s__tuaccept']
		except:
			pass
		# open and satisf auto trades 24 months / number of open auto trades
		try:
			X['eng_au27s_to_au02s'] = X['au27s__tuaccept'] / X['au02s__tuaccept']
		except:
			pass
		# open and satisf auto trades 24 months / number of open satisf trades
		try:
			X['eng_au27s_to_au03s'] = X['au27s__tuaccept'] / X['au03s__tuaccept']
		except:
			pass
		
		#CREDIT CARD TRADES
		# open CC trades vs CC trades
		try:
			X['eng_bc02s_to_bc01s'] = X['bc02s__tuaccept'] / X['bc01s__tuaccept']
		except:
			pass
		# current open satisf CC trades vs CC trades
		try:
			X['eng_bc03s_to_bc01s'] = X['bc03s__tuaccept'] / X['bc01s__tuaccept']
		except:
			pass
		# open CC trades 24m vs CC trades
		try:
			X['eng_bc09s_to_bc01s'] = X['bc09s__tuaccept'] / X['bc01s__tuaccept']
		except:
			pass
		# months since most recent vs month since oldest CC 
		try:	
			X['eng_bc21s_to_bc20s'] = X['bc21s__tuaccept'] / X['bc20s__tuaccept']
		except:
			pass
		# open satisf CC trade 24 months vs CC trades
		try:	
			X['eng_bc27s_to_bc01s'] = X['bc27s__tuaccept'] / X['bc01s__tuaccept']
		except:
			pass
		
		#BANK INSTALLMENTS
		# open bank installment vs bank installment trades
		try:	
			X['eng_bi02s_to_bi01s'] = X['bi02s__tuaccept'] / X['bi01s__tuaccept']
		except:
			pass
		# open satisf bank installment vs number bank installment
		try:	
			X['eng_bi12s_to_bi01s'] = X['bi12s__tuaccept'] / X['bi01s__tuaccept']
		except:
			pass
		# months since most recent bi vs months since oldest bi
		try:	
			X['eng_bi21s_to_bi20s'] = X['bi21s__tuaccept'] / X['bi20s__tuaccept']
		except:
			pass
		# utilization open bi verified 12m vs total open bi verified 12m
		try:
			X['eng_bi34s_to_bi33s'] = X['bi34s__tuaccept'] / X['bi33s__tuaccept']
		except:
			pass
		
		#BANK REVOLVER
		# open br trades vs br trades
		try:	
			X['eng_br02s_to_br01s'] = X['br02s__tuaccept'] / X['br01s__tuaccept']
		except:
			pass
		# open satisf br trades vs br trades
		try:
			X['eng_br03s_to_br01s'] = X['br03s__tuaccept'] / X['br01s__tuaccept']
		except:
			pass
		# br trades opened past 24 / br trades opened
		try:
			X['eng_br09s_to_br01s'] = X['br09s__tuaccept'] / X['br01s__tuaccept']
		except:
			pass
		# months recent br / months oldest br
		try:
			X['eng_br21s_to_br20s'] = X['br21s__tuaccept'] / X['br20s__tuaccept']
		except:
			pass
		# open satis 24m / br trades
		try:
			X['eng_br27s_to_br20s'] = X['br27s__tuaccept'] / X['br20s__tuaccept']
		except:
			pass
		
		#CHARGE OFF TRADES
		# number CO trades in past 24 months / CO trades
		try:
			X['eng_co03s_to_co01s'] = X['co03s__tuaccept'] / X['co01s__tuaccept']
		except:
			pass
		# balance CO 24m / CO balance
		try:
			X['eng_co07s_to_co05s'] = X['co07s__tuaccept'] / X['co05s__tuaccept']
		except:
			pass
		
		# FORECLOSURE TRADES
		# foreclosure trades past 24m / foreclosure trades
		try:
			X['eng_fc03s_to_fc01s'] = X['fc03s__tuaccept'] / X['fc01s__tuaccept']
		except:
			pass
		# balance FC trades 24m / balance FC trades
		try:
			X['eng_fc07s_to_fc05s'] = X['fc07s__tuaccept'] / X['fc05s__tuaccept']
		except:
			pass
		
		#FINANCE INSTALLMENT
		# open fi trades / fi trades
		try:
			X['eng_fi02s_to_fi01s'] = X['fi02s__tuaccept'] / X['fi01s__tuaccept']
		except:
			pass
		# open satisf fi trades / fi trades
		try:
			X['eng_fi03s_to_fi01s'] = X['fi03s__tuaccept'] / X['fi01s__tuaccept']
		except:
			pass
		# number fi opened in past 24 months / opened fi trades
		try:	
			X['eng_fi09s_to_fi02s'] = X['fi09s__tuaccept'] / X['fi02s__tuaccept']
		except:
			pass
		# number fi opened past 24m / number fi trades
		try:
			X['eng_fi09s_to_fi01s'] = X['fi09s__tuaccept'] / X['fi01s__tuaccept']
		except:
			pass
		# months most recent fi opened / months since oldest fi opened
		try:
			X['eng_fi21s_to_fi20s'] = X['fi21s__tuaccept'] / X['fi20s__tuaccept']
		except:
			pass
		# number current open satisf fi 24m / number fi trades
		try:
			X['eng_fi27s_to_fi01s'] = X['fi27s__tuaccept'] / X['fi01s__tuaccept']
		except:
			pass
		
		#FINANCE REVOLVING TRADES
		# number of open FR trades / number FR trades
		try:
			X['eng_fr02s_to_fr01s'] = X['fr02s__tuaccept'] / X['fr01s__tuaccept']
		except:
			pass
		# number current satisf open fr / number FR trades
		try:
			X['eng_fr03s_to_fr01s'] = X['fr03s__tuaccept'] / X['fr01s__tuaccept']
		except:
			pass
		# number opened fr trades 24m / number FR trades
		try:
			X['eng_fr09s_to_fr01s'] = X['fr09s__tuaccept'] / X['fr01s__tuaccept']
		except:
			pass
		
		# HOME EQUITY
		# open home equity vs number of home equity loans
		try:
			X['eng_hi02s_to_hi01s'] = X['hi02s__tuaccept'] / X['hi01s__tuaccept']
		except:
			pass
		# current satisf open he vs number he loans
		try:
			X['eng_hi03s_to_hi01s'] = X['hi03s__tuaccept'] / X['hi01s__tuaccept']
		except:
			pass
		# number he opened past 24m / number he loans
		try:
			X['eng_hi09s_to_hi01s'] = X['hi09s__tuaccept'] / X['hi01s__tuaccept']
		except:
			pass
		# months since most recent he opened / months since oldest
		try:
			X['eng_hi21s_to_hi20s'] = X['hi21s__tuaccept'] / X['hi20s__tuaccept']
		except:
			pass
		# number currently open satisf he loan 24m / number he loans
		try:
			X['eng_hi27s_to_hi01s'] = X['hi27s__tuaccept'] / X['hi01s__tuaccept']
		except:
			pass
		
		# HOME EQUITY LOC
		# number he open LOC / number he LOC
		try:
			X['eng_hr02s_to_hr01s'] = X['hr02s__tuaccept'] / X['hr01s__tuaccept']
		except:
			pass
		# number he opened LOC 24m / number he LOC
		try:
			X['eng_hr12s_to_hr01s'] = X['hr12s__tuaccept'] / X['hr01s__tuaccept']
		except:
			pass
		# months since most recent opened vs months oldest
		try:
			X['eng_hr21s_to_hr20s'] = X['hr21s__tuaccept'] / X['hr20s__tuaccept']
		except:
			pass
		
		# INSTALLMENT TRADES
		# number open installments vs installment trades
		try:
			X['eng_in02s_to_in01s'] = X['in02s__tuaccept'] / X['in01s__tuaccept']
		except:
			pass
		# current open satisf vs installment trades
		try:
			X['eng_in03s_to_in01s'] = X['in03s__tuaccept'] / X['in01s__tuaccept']
		except:
			pass
		# number opened past 24m vs installment trades
		try:
			X['eng_in09s_to_in01s'] = X['in09s__tuaccept'] / X['in01s__tuaccept']
		except:
			pass
		# number open verified in past 12 months vs installment trades
		try:
			X['eng_in12s_to_in01s'] = X['in12s__tuaccept'] / X['in01s__tuaccept']
		except:
			pass
		# months since most recent vs months oldest
		try:
			X['eng_in21s_to_in20s'] = X['in21s__tuaccept'] / X['in20s__tuaccept']
		except:
			pass
		# open satisf 24m vs installment trades
		try:
			X['eng_in27s_to_in01s'] = X['in27s__tuaccept'] / X['in01s__tuaccept']
		except:
			pass
		# open verified 12m vs installment trades
		try:
			X['eng_in28s_to_in01s'] = X['in28s__tuaccept'] / X['in01s__tuaccept']
		except:
			pass
		
		# LOAN MODIFICATIONS
		# number LM mortage 90+DPD vs LM mortgage
		try:
			X['eng_lm08s_to_lm01s'] = X['lm08s__tuaccept'] / X['lm01s__tuaccept']
		except:
			pass
		# bank backed LM vs LM 
		try:
			X['eng_lm25s_to_lm01s'] = X['lm25s__tuaccept'] / X['lm01s__tuaccept']
		except:
			pass
		
		# MORTGAGE TRADES
		# number of open mortgage trades vs number or mortgage trades
		try:
			X['eng_mt02s_to_mt01s'] = X['mt02s__tuaccept'] / X['mt01s__tuaccept']
		except:
			pass
		#number of current satisf MT vs mortgage trades
		try:
			X['eng_mt03s_to_mt01s'] = X['mt03s__tuaccept'] / X['mt01s__tuaccept']
		except:
			pass
		# mt trades opened in 24 months vs mt trades
		try:
			X['eng_mt09s_to_mt01s'] = X['mt09s__tuaccept'] / X['mt01s__tuaccept']
		except:
			pass
		# open verified in past 12 months vs mortgage trades
		try:
			X['eng_mt12s_to_mt01s'] = X['mt12s__tuaccept'] / X['mt01s__tuaccept']
		except:
			pass
		#months most recent opened vs oldest opened
		try:
			X['eng_mt21s_to_mt20s'] = X['mt21s__tuaccept'] / X['mt20s__tuaccept']
		except:
			pass
		# number open satisf MT 24 months vs MT
		try:
			X['eng_mt27s_to_mt01s'] = X['mt27s__tuaccept'] / X['mt01s__tuaccept']
		except:
			pass
		
		# joint trade info
		# trades to joint trades open/satisf
		try:
			X['eng_at03s_to_jt03s'] = X['at03s__tuaccept'] / X['jt03s__tuaccept']
		except:
			pass 
		# auto trades to joint trades open/satisf
		try:
			X['eng_au03s_to_jt03s'] = X['au03s__tuaccept'] / X['jt03s__tuaccept']
		except:
			pass
		# credit card to joint trades open/satisf
		try:
			X['eng_bc03s_to_jt03s'] = X['bc03s__tuaccept'] / X['jt03s__tuaccept']
		except:
			pass
		# bank installments vs JT open/satisf
		try:
			X['eng_bi03s_to_jt03s'] = X['bi03s__tuaccept'] / X['jt03s__tuaccept']
		except:
			pass 
		# bank revolver vs JT open/satisf
		try:
			X['eng_br03s_to_jt03s'] = X['br03s__tuaccept'] / X['jt03s__tuaccept']
		except:
			pass
		# charge offs in 24 months vs JT open/satisf
		try:
			X['eng_co03s_to_jt03s'] = X['co03s__tuaccept'] / X['jt03s__tuaccept']
		except:
			pass
		# foreclosure in 24 months vs JT open/satisf
		try:
			X['eng_fc03s_to_jt03s'] = X['fc03s__tuaccept'] / X['jt03s__tuaccept']
		except:
			pass
		# finance installments vs JT open/satisf
		try:
			X['eng_fi03s_to_jt03s'] = X['fi03s__tuaccept'] / X['jt03s__tuaccept']
		except:
			pass
		# home equity v JT open/satisf
		try:
			X['eng_hi03s_to_jt03s'] = X['hi03s__tuaccept'] / X['jt03s__tuaccept']
		except:
			pass
		# home equity LOC v JT open/satisf
		try:
			X['eng_hr03s_to_jt03s'] = X['hr03s__tuaccept'] / X['jt03s__tuaccept']
		except:
			pass
		# installment trades v JT open/satisf
		try:
			X['eng_in03s_to_jt03s'] = X['in03s__tuaccept'] / X['jt03s__tuaccept']
		except:
			pass
		# mortgage trades vs JT open/satisf
		try:
			X['eng_mt03s_to_jt03s'] = X['mt03s__tuaccept'] / X['jt03s__tuaccept']
		except:
			pass
		
		# joint trades vs individual trades, trades vs application data, bankruptcies, income over trade/ trade over income
		# joint trades
		#-- DONE -- X['jt01s__tuaccept'] total joint trades  test against auto, credit card, bi, br, CO, Foreclosure, home equity, installments
		#-- DONE -- X['jt03s__tuaccept'] number open + satisfactory
		# X['jt21s__tuaccept'] / X['jt20s__tuaccept'] months most recent / months oldest
		
		#income and tradeline info
		# 32s 33s 34s 35s max, total, utilization, average verified 12 months vs income
		# ================================== trades over income ==============================================
		# try:
		# 	X['eng_at32s_to_income'] = X['at32s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_at33a_to_income'] = X['at33a__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_at33b_to_income'] = X['at33b__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_at34a_to_income'] = X['at34a__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_at34b_to_income'] = X['at34b__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_at35a_to_income'] = X['at35a__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_at35b_to_income'] = X['at35b__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# ============================== income over trades ============================================
		try:
			X['eng_income_to_at32s'] = X['fltgrossmonthly__income_sum'] / X['at32s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_at33a'] = X['fltgrossmonthly__income_sum'] / X['at33a__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_at33b'] = X['fltgrossmonthly__income_sum'] / X['at33b__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_at34a'] = X['fltgrossmonthly__income_sum'] / X['at34a__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_at34b'] = X['fltgrossmonthly__income_sum'] / X['at34b__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_at35a'] = X['fltgrossmonthly__income_sum'] / X['at35a__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_at35b'] = X['fltgrossmonthly__income_sum'] / X['at35b__tuaccept']
		except:
			pass

		#================================= auto trades over income ========================================
		# try:
		# 	X['eng_au32s_to_income'] = X['au32s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_au33s_to_income'] = X['au33s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_au34s_to_income'] = X['au34s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_au35s_to_income'] = X['au35s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass

		# =============================== income over auto trades =========================================
		try:
			X['eng_income_to_au32s'] = X['fltgrossmonthly__income_sum'] / X['au32s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_au33s'] = X['fltgrossmonthly__income_sum'] / X['au33s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_au34s'] = X['fltgrossmonthly__income_sum'] / X['au34s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_au35s'] = X['fltgrossmonthly__income_sum'] / X['au35s__tuaccept']
		except:
			pass		
		
		#=============================== credit cards over income =========================================
		# try:
		# 	X['eng_bc32s_to_income'] = X['bc32s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_bc33s_to_income'] = X['bc33s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_bc34s_to_income'] = X['bc34s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_bc35s_to_income'] = X['bc35s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# ============================== income over credit cards =========================================
		try:
			X['eng_income_to_bc32s'] = X['fltgrossmonthly__income_sum'] / X['bc32s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_bc33s'] = X['fltgrossmonthly__income_sum'] / X['bc33s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_bc34s'] = X['fltgrossmonthly__income_sum'] / X['bc34s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_bc35s'] = X['fltgrossmonthly__income_sum'] / X['bc35s__tuaccept']
		except:
			pass


		#=============================== bank installments over income====================================
		# try:
		# 	X['eng_bi32s_to_income'] = X['bi32s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_bi33s_to_income'] = X['bi33s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_bi34s_to_income'] = X['bi34s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_bi35s_to_income'] = X['bi35s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		
		#============================= income over bank installments ====================================
		try:
			X['eng_income_to_bi32s'] = X['fltgrossmonthly__income_sum'] / X['bi32s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_bi33s'] = X['fltgrossmonthly__income_sum'] / X['bi33s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_bi34s'] = X['fltgrossmonthly__income_sum'] / X['bi34s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_bi35s'] = X['fltgrossmonthly__income_sum'] / X['bi35s__tuaccept']
		except:
			pass

		#============================= bank revolvers over income========================================
		# try:
		# 	X['eng_br32s_to_income'] = X['br32s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_br33s_to_income'] = X['br33s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_br34s_to_income'] = X['br34s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_br35s_to_income'] = X['br35s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		
		#============================ income over bank revolvers ========================================
		try:
			X['eng_income_to_br32s'] = X['fltgrossmonthly__income_sum'] / X['br32s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_br33s'] = X['fltgrossmonthly__income_sum'] / X['br33s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_br34s'] = X['fltgrossmonthly__income_sum'] / X['br34s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_br35s'] = X['fltgrossmonthly__income_sum'] / X['br35s__tuaccept']
		except:
			pass

		#============================ finance installments over income===================================
		# try:
		# 	X['eng_fi32s_to_income'] = X['fi32s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_fi33s_to_income'] = X['fi33s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_fi34s_to_income'] = X['fi34s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_fi35s_to_income'] = X['fi35s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		
		# =========================== income over finance installments ==================================
		try:
			X['eng_income_to_fi32s'] = X['fltgrossmonthly__income_sum'] / X['fi32s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_fi33s'] = X['fltgrossmonthly__income_sum'] / X['fi33s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_fi34s'] = X['fltgrossmonthly__income_sum'] / X['fi34s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_fi35s'] = X['fltgrossmonthly__income_sum'] / X['fi35s__tuaccept']
		except:
			pass

		#============================ finance revolvers over income======================================
		# try:
		# 	X['eng_fr32s_to_income'] = X['fr32s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_fr33s_to_income'] = X['fr33s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_fr34s_to_income'] = X['fr34s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_fr35s_to_income'] = X['fr35s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		#============================ income over finance revolvers =====================================
		try:
			X['eng_income_to_fr32s'] = X['fltgrossmonthly__income_sum'] / X['fr32s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_fr33s'] = X['fltgrossmonthly__income_sum'] / X['fr33s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_fr34s'] = X['fltgrossmonthly__income_sum'] / X['fr34s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_fr35s'] = X['fltgrossmonthly__income_sum'] / X['fr35s__tuaccept']
		except:
			pass

		#============================ home equity over income============================================
		# try:
		# 	X['eng_hi32s_to_income'] = X['hi32s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_hi33s_to_income'] = X['hi33s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_hi34s_to_income'] = X['hi34s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_hi35s_to_income'] = X['hi35s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		
		#=============================income over home equity===========================================
		try:
			X['eng_income_to_hi32s'] = X['fltgrossmonthly__income_sum'] / X['hi32s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_hi33s'] = X['fltgrossmonthly__income_sum'] / X['hi33s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_hi34s'] = X['fltgrossmonthly__income_sum'] / X['hi34s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_hi35s'] = X['fltgrossmonthly__income_sum'] / X['hi35s__tuaccept']
		except:
			pass

		#========================== HE LOC over income==================================================
# 		try:
# 			X['eng_hr32s_to_income'] = X['hr32s__tuaccept'] / X['fltgrossmonthly__income_sum']
# 		except:
# 			pass
# # 		try:
# # 			X['eng_hr33s_to_income'] = X['hr33s__tuaccept'] / X['fltgrossmonthly__income_sum']
# # 		except:
# # 			pass
# 		try:
# 			X['eng_hr34s_to_income'] = X['hr34s__tuaccept'] / X['fltgrossmonthly__income_sum']
# 		except:
# 			pass
# # 		try:
# # 			X['eng_hr35s_to_income'] = X['hr35s__tuaccept'] / X['fltgrossmonthly__income_sum']
# # 		except:
# # 			pass
		
		#============================ income over HE LOC ===============================================
		try:
			X['eng_income_to_hr32s'] = X['fltgrossmonthly__income_sum'] / X['hr32s__tuaccept']
		except:
			pass
# 		try:
# 			X['eng_income_to_hr33s'] = X['fltgrossmonthly__income_sum'] / X['hr33s__tuaccept']
# 		except:
# 			pass
		try:
			X['eng_income_to_hr34s'] = X['fltgrossmonthly__income_sum'] / X['hr34s__tuaccept']
		except:
			pass
# 		try:
# 			X['eng_income_to_hr35s'] = X['fltgrossmonthly__income_sum'] / X['hr35s__tuaccept']
# 		except:
# 			pass

		#============================ installment trades over income=====================================
		# try:
		# 	X['eng_in32s_to_income'] = X['in32s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_in33s_to_income'] = X['in33s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_in34s_to_income'] = X['in34s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_in35s_to_income'] = X['in35s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		
		#=========================== income over installment trades =====================================
		try:
			X['eng_income_to_in32s'] = X['fltgrossmonthly__income_sum'] / X['in32s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_in33s'] = X['fltgrossmonthly__income_sum'] / X['in33s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_in34s'] = X['fltgrossmonthly__income_sum'] / X['in34s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_in35s'] = X['fltgrossmonthly__income_sum'] / X['in35s__tuaccept']
		except:
			pass
		

		#=========================== mortgage trades over income=========================================
		# try:
		# 	X['eng_mt32s_to_income'] = X['mt32s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_mt33s_to_income'] = X['mt33s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_mt34s_to_income'] = X['mt34s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass
		# try:
		# 	X['eng_mt35s_to_income'] = X['mt35s__tuaccept'] / X['fltgrossmonthly__income_sum']
		# except:
		# 	pass

		#============================ income over mortgage trades ========================================
		try:
			X['eng_income_to_mt32s'] = X['fltgrossmonthly__income_sum'] / X['mt32s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_mt33s'] = X['fltgrossmonthly__income_sum'] / X['mt33s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_mt34s'] = X['fltgrossmonthly__income_sum'] / X['mt34s__tuaccept']
		except:
			pass
		try:
			X['eng_income_to_mt35s'] = X['fltgrossmonthly__income_sum'] / X['mt35s__tuaccept']
		except:
			pass

		#======================== Bankruptcy fields =========================================
		#bankruptcy 24 months vs bankruptcy
		try:
			X['eng_g099s_to_g094s'] = X['g099s__tuaccept'] / X['g094s__tuaccept']
		except:
			pass
		#number of trade bankruptcies vs number of trades
		try:
			X['eng_g100s_to_at01s'] = X['g100s__tuaccept'] / X['at01s__tuaccept']
		except:
			pass
		#number of trade bankruptcies vs number of open trades
		try:
			X['eng_g100s_to_at02s'] = X['g100s__tuaccept'] / X['at02s__tuaccept']
		except:
			pass
		#number of trade bankruptcies vs number of open/satisf trades
		try:
			X['eng_g100s_to_at03s'] = X['g100s__tuaccept'] / X['at03s__tuaccept']
		except:
			pass
		#number of trade bankruptcies verified in past 24 months vs trades opened in past 24
		try:
			X['eng_g099a_to_at09s'] = X['g099a__tuaccept'] / X['at09s__tuaccept']
		except:
			pass
		#number of trade bankruptcies verified in past 24 months vs open satisf trades 24
		try:
			X['eng_g099a_to_at27s'] = X['g099a__tuaccept'] / X['at27s__tuaccept']
		except:
			pass

		# BK fields
		# --DONE--X['g094s__tuaccept'] number of public record bankruptcies
		# --DONE--X['g099s__tuaccept'] number of public BK past 24 months
		# --DONE--X['g100s__tuaccept'] number of tradeline BK
		# try:
		# 	X['eng_inttype_times_linkf032'] = X['intType__app'] * X['linkf032__tucvlink']
		# except:
		# 	pass
		# # months since most recent inquiry over months since most recent public record BK
		# try:
		# 	X['eng_g102s_to_s207s'] = X['g102s__tuaccept'] / X['s207s__tuaccept']
		# except:
		# 	pass
		# # months since most recent inquiry over months since most recent tradeline BK
		# try:
		# 	X['eng_g102s_to_s207a'] = X['g102s__tuaccept'] / X['s207a__tuaccept']
		# except:
		# 	pass
		# =====================BK between CREDIT VISION VS FACTOR TRUST======================
		# public record BK 24 months
		try:
			X['eng_linkt008_times_g099s'] = X['linkt008__tucvlink'] * X['g099s__tuaccept']
		except:
			pass
		# trade line BK
		try:
			X['eng_linkt009_times_g100s'] = X['linkt009__tucvlink'] * X['g100s__tuaccept']
		except:
			pass
		# public record BK	
		try:
			X['eng_linkt010_times_g094s'] = X['linkt010__tucvlink'] * X['g094s__tuaccept']
		except:
			pass
		# bankruptcy count 24 months vs bk count total=========================================
		try:
			X['eng_bk24_to_bk_ln'] = X['bankruptcycount24month__ln'] / X['bankruptcycount__ln']
		except: 
			pass
		# derogatory count 12 months vs derog count total
		try:
			X['eng_derog12_to_derog_ln'] = X['derogcount12month__ln'] / X['derogcount__ln']
		except:
			pass 
		# eviction counts 12 months vs eviction total
		try:
			X['eng_evict12_to_evict_ln'] = X['evictioncount12month__ln'] / X['evictioncount__ln']
		except: 
			pass
		# source credit header newest vs olderst
		try:
			X['eng_sch_new_to_old'] = X['sourcecredheadertimenewest__ln'] / X['sourcecredheadertimeoldest__ln']
		except:
			pass
		# source non-derog count newest vs oldest
		try:
			X['eng_snd_new_to_old'] = X['sourcenonderogcount12month__ln'] / X['sourcenonderogcount__ln']
		except:
			pass
		# real estate asset purchase time newest vs oldest
		try:
			X['eng_assetproppurchase_new_to_old'] = X['assetproppurchasetimenewest__ln'] / X['assetproppurchasetimeoldest__ln']
		except:
			pass
		# real estate asset sale time newest vs oldest
		try:
			X['eng_assetpropsale_new_to_old'] = X['assetpropsaletimenewest__ln'] / X['assetpropsaletimeoldest__ln']
		except:
			pass
		# lien judgement foreclosure count 12 mo vs count total
		try:
			X['eng_ljcount_12m_to_total'] = X['lienjudgmentcount12month__ln'] / X['lienjudgmentcount__ln']
		except:
			pass
		# short term loan request 24months vs total
		try:
			X['eng_stlr_24mo_to_total'] = X['shorttermloanrequest24month__ln'] / X['shorttermloanrequest__ln']
		except:
			pass
		# return
		return X

list_cols_eng_raw = ['fltamountfinanced__app', 'fltapprovedpricewholesale__app', 'jt03s__tuaccept',
					 'fltgrossmonthly__income_sum', 'fltapproveddowntotal__app', 'fltamountfinanced__app', 'fltapproveddowntotal__app', 
					 'fltgrossmonthly__income_sum', 'fltgrossmonthly__income_std', 'fltgrossmonthly__income_min', 'fltgrossmonthly__income_median',
					 'fltgrossmonthly__income_mean', 'fltgrossmonthly__income_max', 'fltapproveddowntotal__app', 'fltapprovedpricewholesale__app', 'at02s__tuaccept', 
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
					 'mt34s__tuaccept', 'mt35s__tuaccept', 'g099s__tuaccept', 'intType__app', 'linkf032__tucvlink__bin', 'linkt008__tucvlink',
					 'linkt009__tucvlink', 'linkt010__tucvlink', 'g094s__tuaccept', 'bankruptcycount24month__ln', 'bankruptcycount__ln',
					 'derogcount12month__ln', 'derogcount__ln', 'sourcecredheadertimenewest__ln', 'sourcecredheadertimeoldest__ln',
					 'sourcenonderogcount12month__ln', 'sourcenonderogcount__ln', 'assetproppurchasetimenewest__ln', 'assetproppurchasetimeoldest__ln',
					 'assetpropsaletimenewest__ln', 'assetpropsaletimeoldest__ln', 'lienjudgmentcount12month__ln', 'lienjudgmentcount__ln',
					 'shorttermloanrequest24month__ln', 'shorttermloanrequest__ln','evictioncount12month__ln', 'evictioncount__ln']

