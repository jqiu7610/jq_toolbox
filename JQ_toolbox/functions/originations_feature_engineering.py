import numpy as np
import datetime as dt



class Feature_Engineering_JQ:
	# transform
	def transform(self, X):
		# LN 
		# ratio BK within 24 months
		try:
			X['ratio_bk_within_24months__eng'] = X['fltBankruptcyCount24Month__ln'] / X['fltBankruptcyCount__ln']
		except:
			pass
		# try amount financed vs price of the vehicle
		try:
			X['ratio_amtfin_to_price__eng'] = X['fltApprovedAmountFinanced__data'] / X['fltApprovedSalesPrice__data']
		except:
			pass
		# down to amt financed
		try:
			X['ratio_down_to_amtfin__eng'] = X['fltApprovedDownTotal__data'] / X['fltApprovedAmountFinanced__data']
		except:
			pass
		
		# application days
		try:
			X['days_between_app_approval__eng'] = X['DaysSinceApplication__data'] - X['DaysSinceApproval__data']
		except:
			pass
		# non-derog count ratio within 3 months
		try:
			X['nonderog_count_ratio_3mo__eng'] = X['fltSourceNonDerogCount03Month__ln'] / X['fltSourceNonDerogCount__ln']
		except:
			pass
		# non-derog count ratio within 6 months
		try:
			X['nonderog_count_ratio_6mo__eng'] = X['fltSourceNonDerogCount06Month__ln'] / X['fltSourceNonDerogCount__ln']
		except:
			pass
		# non-derog count ratio within 12 months
		try:
			X['nonderog_count_ratio_12mo__eng'] = X['fltSourceNonDerogCount12Month__ln'] / X['fltSourceNonDerogCount__ln']
		except:
			pass
		# derog count ratio within 12 months
		try:
			X['derog_count_ratio_12mo__eng'] = X['fltDerogCount12Month__ln'] / X['fltDerogCount__ln']
		except:
			pass
		# crim felon ratio within 12 months
		try:
			X['crim_felon_ratio_12mo__eng'] = X['fltCriminalFelonyCount12Month__ln'] / X['fltCriminalFelonyCount__ln']
		except:
			pass
		# crim nonfelon ratio within 12 months
		try:
			X['crim_nonfelon_ratio_12mo__eng'] = X['fltCriminalNonFelonyCount12Month__ln'] / X['fltCriminalNonFelonyCount__ln']
		except:
			pass
		# lien judge count ratio within 12 months
		try:
			X['lien_judge_ratio_12mo__eng'] = X['fltLienJudgmentCount12Month__ln'] / X['fltLienJudgmentCount__ln']
		except:
			pass
		# income to various flt fields
		# income to down
		try:
			X['income_to_pmt__eng'] = X['fltGrossMonthly__income_sum'] / X['fltApprovedPayment__data']
		except:
			pass
		try:
			X['income_to_down__eng'] = X['fltGrossMonthly__income_sum'] / X['fltApprovedDownCash__data']
		except:
			pass
		try:
 			X['income_to_down2__eng'] = X['fltGrossMonthly__income_sum'] / X['fltDownCash__data']
		except:
 			pass
		 # income to total payments
		try:
 			X['income_to_totpmt__eng'] = X['fltGrossMonthly__income_sum'] / X['fltTotalPayment__data']
		except:
 			pass
		# income to sale price
		try:
			X['income_to_sale__eng'] = X['fltGrossMonthly__income_sum'] / X['fltApprovedSalesPrice__data']
		except:
			pass
		# income to amt financed
		try:
			X['income_to_amtfin__eng'] = X['fltGrossMonthly__income_sum'] / X['fltAmountFinanced__data']
		except:
			pass
		try:
			X['income_to_months_at__eng'] = X['fltGrossMonthly__income_sum'] / X['MonthsAtIncome__income_sum']
		except:
			pass
		try:
			X['income_to_months_mean_at__eng'] = X['fltGrossMonthly__income_mean'] / X['MonthsAtIncome__income_mean']
		except:
			pass
		try:
			X['income_to_months_med_at__eng'] = X['fltGrossMonthly__income_median'] / X['MonthsAtIncome__income_median']
		except:
			pass
		try:
			X['income_to_months_max_at__eng'] = X['fltGrossMonthly__income_max'] / X['MonthsAtIncome__income_max']
		except:
			pass
		try:
			X['income_to_months_min_at__eng'] = X['fltGrossMonthly__income_min'] / X['MonthsAtIncome__income_min']
		except:
			pass

		try:
			X['income_over_sources__eng'] = X['fltGrossMonthly__income_sum'] / X['fltGrossMonthly__income_nunique']
		except:
			pass
		try:
			X['Months_at_over_nunique__eng'] = X['MonthsAtIncome__income_sum'] / X['MonthsAtIncome__income_nunique']
		except:
			pass
		try:
			X['balance_current_over_original__eng'] = X['fltBalanceCurrent__debt_sum'] / X['fltBalanceOriginal__debt_sum']
		except:
			pass
		try:
			X['balance_current_over_original_mean__eng'] = X['fltBalanceCurrent__debt_mean'] / X['fltBalanceOriginal__debt_mean']
		except:
			pass
		try:
			X['balance_current_over_original_median_eng'] = X['fltBalanceCurrent__debt_median'] / X['fltBalanceOriginal__debt_median']
		except:
			pass
		try:
			X['balance_current_over_original_max_eng'] = X['fltBalanceCurrent__debt_max'] / X['fltBalanceOriginal__debt_max']
		except:
			pass
		try:
			X['balance_current_over_original_min_eng'] = X['fltBalanceCurrent__debt_min'] / X['fltBalanceOriginal__debt_min']
		except:
			pass
		try:
			X['income_over_debtpmt__eng'] = X['fltGrossMonthly__income_sum'] / X['fltMonthlyPayment__debt_sum']
		except:
			pass
		try:
			X['income_over_debtpmt_mean_eng'] = X['fltGrossMonthly__income_mean'] / X['fltMonthlyPayment__debt_mean']
		except:
			pass
		try:
			X['income_over_debtpmt_median_eng'] = X['fltGrossMonthly__income_median'] / X['fltMonthlyPayment__debt_median']
		except:
			pass
		try:
			X['income_over_debtpmt_max_eng'] = X['fltGrossMonthly__income_max'] / X['fltMonthlyPayment__debt_max']
		except:
			pass
		try:
			X['income_over_debtpmt_min_eng'] = X['fltGrossMonthly__income_min'] / X['fltMonthlyPayment__debt_min']
		except:
			pass
		try:
			X['record_time_oldest_to_newest__eng'] = X['fltSubjectRecordTimeOldest__ln'] - X['fltSubjectRecordTimeNewest__ln']
		except:
			pass
		try:
			X['header_oldest_to_newest__eng'] = X['fltSourcecredHeaderTimeOldest__ln'] - X['fltSourcecredHeaderTimeNewest__ln']
		except:
			pass
		try:
			X['prop_purchase_oldest_to_newest__eng'] = X['fltAssetProppurchaseTimeOldest__ln'] - X['fltAssetPropeversoldCount__ln']
		except:
			pass
		try:
			X['prop_sold_12mo_to_tot__eng'] = X['fltAssetPropsoldCount12Month__ln'] / X['fltAssetProppurchaseTimeNewest__ln']
		except:
			pass
		try:
			X['prop_sale_oldest_to_newest__eng'] = X['fltAssetPropsaleTimeOldest__ln'] - X['fltAssetPropsaleTimeNewest__ln']
		except:
			pass
		try:
			X['addr_input_old_to_new__eng'] = X['fltAddrInputTimeOldest__ln'] - X['fltAddrInputTimeNewest__ln']
		except:
			pass
		try:
			X['addr_current_old_to_new__eng'] = X['fltAddrCurrentTimeOldest__ln'] - X['fltAddrCurrentTimeNewest__ln']
		except:
			pass
		try:
			X['eviction_12mo_ratio__eng'] = X['fltEvictionCount12Month__ln'] / X['fltEvictionCount__ln']
		except:
			pass
		try:
			X['lien_12mo_ratio__eng'] = X['fltLienJudgmentCount12Month__ln'] / X['fltLienJudgmentCount__ln']
		except:
			pass
		try:
			X['shortterm_loan_12mo_ratio__eng'] = X['fltShortTermLoanRequest12Month__ln'] / X['fltShortTermLoanRequest__ln']
		except:
			pass
		try:
			X['shortterm_loan_24mo_ratio__eng'] = X['fltShortTermLoanRequest24Month__ln'] / X['fltShortTermLoanRequest__ln']
		except:
			pass
		try:
			X['shortterm_auto_12mo_ratio__eng'] = X['fltInquiryAuto12Month__ln'] / X['fltShortTermLoanRequest__ln']
		except:
			pass
		try:
			X['shortterm_bank_12mo_ratio__eng'] = X['fltInquiryBanking12Month__ln'] / X['fltShortTermLoanRequest__ln']
		except:
			pass
		try:
			X['shortterm_tele_12mo_ratio__eng'] = X['fltInquiryTelcom12Month__ln'] / X['fltShortTermLoanRequest__ln']
		except:
			pass

		# Quarter relative to year
		# sin
		try:
			X['date_quarter_year_sin__cyclic'] = np.sin((X['date'].dt.quarter-1) * (2*np.pi/4))
		except:
			pass
		# cos
		try:
			X['date_quarter_year_cos__cyclic'] = np.cos((X['date'].dt.quarter-1) * (2*np.pi/4))
		except:
			pass
		# tan
		try:
			X['date_quarter_year_tan__cyclic'] = X['date_quarter_year_sin__cyclic'] / X['date_quarter_year_cos__cyclic']
		except:
			pass

		# Month relative to year
		# sin
		try:
			X['date_month_year_sin__cyclic'] = np.sin((X['date'].dt.month-1) * (2*np.pi/12))
		except:
			pass
		# cos
		try:
			X['date_month_year_cos__cyclic'] = np.cos((X['date'].dt.month-1) * (2*np.pi/12))
		except:
			pass
		# tan
		try:
			X['date_month_year_tan__cyclic'] = X['date_month_year_sin__cyclic'] / X['date_month_year_cos__cyclic']
		except:
			pass

		# Day relative to week
		# sin
		try:
			X['date_day_week_sin__cyclic'] = np.sin((X['date'].dt.dayofweek-1) * (2*np.pi/7))
		except:
			pass
		# cos
		try:
			X['date_day_week_cos__cyclic'] = np.cos((X['date'].dt.dayofweek-1) * (2*np.pi/7))
		except:
			pass
		# tan
		try:
			X['date_day_week_tan__cyclic'] = X['date_day_week_sin__cyclic'] / X['date_day_week_cos__cyclic']
		except:
			pass

		# Day relative to month
		# sin
		try:
			X['date_day_month_sin__cyclic'] = np.sin((X['date'].dt.day-1) * (2*np.pi/X['date'].dt.daysinmonth))
		except:
			pass
		# cos
		try:
			X['date_day_month_cos__cyclic'] = np.cos((X['date'].dt.day-1) * (2*np.pi/X['date'].dt.daysinmonth))
		except:
			pass
		# tan
		try:
			X['date_day_month_tan__cyclic'] = X['date_day_month_sin__cyclic'] / X['date_day_month_cos__cyclic']
		except:
			pass

		# Day relative to year
		# sin
		try:
			X['date_day_year_sin__cyclic'] = np.sin((X['date'].dt.dayofyear-1) * (2*np.pi/365))
		except:
			pass
		# cos
		try:
			X['date_day_year_cos__cyclic'] = np.cos((X['date'].dt.dayofyear-1) * (2*np.pi/365))
		except:
			pass
		# tan
		try:
			X['date_day_year_tan__cyclic'] = X['date_day_year_sin__cyclic'] / X['date_day_year_cos__cyclic']
		except:
			pass
		
		#--------------------------------------------------------------------------
		# LN Dates
		#--------------------------------------------------------------------------
		# Quarter relative to year
		# sin
		try:
			X['date_quarter_year_sin_ln_cyclic'] = np.sin((X['dtmStampCreation__ln'].dt.quarter-1) * (2*np.pi/4))
		except:
			pass
		# cos
		try:
			X['date_quarter_year_cos_ln_cyclic'] = np.cos((X['dtmStampCreation__ln'].dt.quarter-1) * (2*np.pi/4))
		except:
			pass
		# tan
		try:
			X['date_quarter_year_tan_ln_cyclic'] = X['date_quarter_year_sin_ln_cyclic'] / X['date_quarter_year_cos_ln_cyclic']
		except:
			pass

		# Month relative to year
		# sin
		try:
			X['date_month_year_sin_ln_cyclic'] = np.sin((X['dtmStampCreation__ln'].dt.month-1) * (2*np.pi/12))
		except:
			pass
		# cos
		try:
			X['date_month_year_cos_ln_cyclic'] = np.cos((X['dtmStampCreation__ln'].dt.month-1) * (2*np.pi/12))
		except:
			pass
		# tan
		try:
			X['date_month_year_tan_ln_cyclic'] = X['date_month_year_sin_ln_cyclic'] / X['date_month_year_cos_ln_cyclic']
		except:
			pass

		# Day relative to week
		# sin
		try:
			X['date_day_week_sin_ln_cyclic'] = np.sin((X['dtmStampCreation__ln'].dt.dayofweek-1) * (2*np.pi/7))
		except:
			pass
		# cos
		try:
			X['date_day_week_cos_ln_cyclic'] = np.cos((X['dtmStampCreation__ln'].dt.dayofweek-1) * (2*np.pi/7))
		except:
			pass
		# tan
		try:
			X['date_day_week_tan_ln_cyclic'] = X['date_day_week_sin_ln_cyclic'] / X['date_day_week_cos_ln_cyclic']
		except:
			pass

		# Day relative to month
		# sin
		try:
			X['date_day_month_sin_ln_cyclic'] = np.sin((X['dtmStampCreation__ln'].dt.day-1) * (2*np.pi/X['dtmStampCreation__ln'].dt.daysinmonth))
		except:
			pass
		# cos
		try:
			X['date_day_month_cos_ln_cyclic'] = np.cos((X['dtmStampCreation__ln'].dt.day-1) * (2*np.pi/X['dtmStampCreation__ln'].dt.daysinmonth))
		except:
			pass
		# tan
		try:
			X['date_day_month_tan_ln_cyclic'] = X['date_day_month_sin_ln_cyclic'] / X['date_day_month_cos_ln_cyclic']
		except:
			pass

		# Day relative to year
		# sin
		try:
			X['date_day_year_sin_ln_cyclic'] = np.sin((X['dtmStampCreation__ln'].dt.dayofyear-1) * (2*np.pi/365))
		except:
			pass
		# cos
		try:
			X['date_day_year_cos_ln_cyclic'] = np.cos((X['dtmStampCreation__ln'].dt.dayofyear-1) * (2*np.pi/365))
		except:
			pass
		# tan
		try:
			X['date_day_year_tan_ln_cyclic'] = X['date_day_year_sin_ln_cyclic'] / X['date_day_year_cos_ln_cyclic']
		except:
			pass
		# return
		return X