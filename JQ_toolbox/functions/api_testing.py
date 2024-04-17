import ast
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# define class
class TimeParsing:
	# initialize
	def __init__(self, cls_parse_payload, df_payloads):
		self.cls_parse_payload = cls_parse_payload
		self.df_payloads = df_payloads
	# parse
	def parse_payloads(self):
		# counter offers
		list_X_lg_grouped = []
		# parsed X
		list_x_parsed = []
		# missing, errors, reasons
		list_dict_n_miss = []
		list_list_errors_flat = []
		list_list_reasons_flat = []
		# times
		list_flt_sec = [] # overall
		list_flt_sec_get_payloads = [] # get payloads
		list_flt_sec_parse = [] # parse
		list_flt_sec_create_x = [] # create x
		list_flt_sec_preprocessing = [] # preprocess
		list_flt_sec_counter = [] # counter offers
		#list_flt_sec_predict = [] # predict
		list_flt_sec_adv_act = [] # adverse action
		list_flt_sec_gen_output = [] # generate output
		# n debtors
		list_int_n_debtors = []
		# list of error indices
		list_idx_errors = []
		# get n rows
		int_n_rows = self.df_payloads.shape[0]
		# iterate
		for a in range(int_n_rows):
			# print message
			print(f'Payload {a+1}/{int_n_rows}')
			# get json_str_request
			json_str_request = self.df_payloads['strZestRequest'].iloc[a]
			# convert to dict
			json_str_request = ast.literal_eval(json_str_request)
			# start timer
			time_start = time.perf_counter()
			# generate output
			self.cls_parse_payload.generate_output(json_str_request=json_str_request)
			# get time in sec
			flt_sec = time.perf_counter() - time_start
			# get parsed x
			X_parsed = self.cls_parse_payload.X
			# get n debtors
			int_n_debtors = X_parsed.shape[0]
			# place identifiers
			X_parsed['bigAccountId'] = [self.df_payloads['bigAccountId'].iloc[a] for i in range(int_n_debtors)]
			X_parsed['bigZestV2Id'] = [self.df_payloads['bigZestV2Id'].iloc[a] for i in range(int_n_debtors)]
			X_parsed['dtmCreatedDate'] = [self.df_payloads['dtmCreatedDate'].iloc[a] for i in range(int_n_debtors)]
			# place predictions
			X_parsed['y_hat_pd'] = list(self.cls_parse_payload.y_hat_pd)
			X_parsed['y_hat_lgd'] = list(self.cls_parse_payload.y_hat_lgd)
			# append to lists
			# list_X_lg_grouped
			list_X_lg_grouped.append(self.cls_parse_payload.X_lg_grouped_pre_sub)
			# X_parsed
			list_x_parsed.append(X_parsed)
			# missing, errors, reasons
			list_dict_n_miss.append(self.cls_parse_payload.dict_n_miss)
			list_list_errors_flat.append([item for sublist in self.cls_parse_payload.list_list_errors for item in sublist])
			list_list_reasons_flat.append([item for sublist in self.cls_parse_payload.list_list_reasons for item in sublist])
			# times
			list_flt_sec.append(flt_sec) # overall
			list_flt_sec_get_payloads.append(self.cls_parse_payload.flt_sec_get_payloads) # get payloads
			list_flt_sec_parse.append(self.cls_parse_payload.flt_sec_parse) # parse
			list_flt_sec_create_x.append(self.cls_parse_payload.flt_sec_create_x) # create x
			list_flt_sec_preprocessing.append(self.cls_parse_payload.flt_sec_preprocessing) # preprocess
			list_flt_sec_counter.append(self.cls_parse_payload.flt_sec_counter) # counter offers
			#list_flt_sec_predict.append(self.cls_parse_payload.flt_sec_predict) # predict
			list_flt_sec_adv_act.append(self.cls_parse_payload.flt_sec_adv_act) # adverse action
			list_flt_sec_gen_output.append(self.cls_parse_payload.flt_sec_gen_output) # generate output
			# n debtors
			list_int_n_debtors.append(int_n_debtors)
			# print current mean time
			print(f'Mean parsing time: {np.mean(list_flt_sec):0.5}')
		# create df
		df_output = pd.DataFrame({'counter_offers': list_X_lg_grouped,
								  'x_parsed': list_x_parsed,
								  'n_miss': list_dict_n_miss,
								  'errors': list_list_errors_flat,
								  'reasons': list_list_reasons_flat,
								  'sec': list_flt_sec,
								  'sec_get_payloads': list_flt_sec_get_payloads,
								  'sec_parse': list_flt_sec_parse,
								  'sec_create_x': list_flt_sec_create_x,
								  'sec_preprocessing': list_flt_sec_preprocessing,
								  'sec_counter': list_flt_sec_counter,
								  #'sec_predict': list_flt_sec_predict,
								  'sec_adv_act': list_flt_sec_adv_act,
								  'sec_gen_output': list_flt_sec_gen_output,
			                      'n_debtors': list_int_n_debtors})
		# save to object
		self.list_idx_errors = list_idx_errors
		self.df_output = df_output
		# return
		return self
	# plot
	def create_plot(self, tpl_figsize, str_filename):
		# ax
		fig, ax = plt.subplots(nrows=5, ncols=1, figsize=tpl_figsize)
		# altogether
		flt_mean_all = np.mean(self.df_output['sec'])
		ax[0].set_title(f"All Debtors (mean = {flt_mean_all:0.5} sec; N = {self.df_output.shape[0]})")
		sns.distplot(self.df_output['sec'], kde=True, ax=ax[0])
		# 1 debtor
		df_output_1 = self.df_output[self.df_output['n_debtors']==1]
		flt_mean_1 = np.mean(df_output_1['sec'])
		ax[1].set_title(f"One Debtor (mean = {flt_mean_1:0.5} sec; N = {df_output_1.shape[0]})")
		sns.distplot(df_output_1['sec'], kde=True, ax=ax[1])
		# 2 debtors
		df_output_2 = self.df_output[self.df_output['n_debtors']==2]
		flt_mean_2 = np.mean(df_output_2['sec'])
		ax[2].set_title(f"Two Debtors (mean = {flt_mean_2:0.5} sec; N = {df_output_2.shape[0]})")
		sns.distplot(df_output_2['sec'], kde=True, ax=ax[2])
		# bar plot
		ax[3].set_title('Mean Seconds by N Debtors')
		ax[3].bar(['All', '1 Debtor', '2 Debtors'], [flt_mean_all, flt_mean_1, flt_mean_2])
		# create df for grouped bar plot
		list_int_n_debtors = []
		list_str_step_name = []
		list_flt_sec_mean = []
		for int_n_debtors in [1,2]:
			for str_step_name in ['sec_get_payloads','sec_parse','sec_create_x','sec_preprocessing','sec_counter','sec_adv_act','sec_gen_output']:
				# subset
				df_tmp = self.df_output[self.df_output['n_debtors']==int_n_debtors]
				# get mean of step
				flt_sec_mean = np.mean(df_tmp[str_step_name])
				# append to lists
				list_int_n_debtors.append(int_n_debtors)
				list_str_step_name.append(str_step_name)
				list_flt_sec_mean.append(flt_sec_mean)
		# make df
		df_for_plot = pd.DataFrame({'N Debtors': list_int_n_debtors,
			                        'Steps': list_str_step_name,
			                        'Mean Seconds': list_flt_sec_mean})
		# dictionary for mapping steps
		dict_map_steps = {'sec_get_payloads': 'Get Payloads',
						  'sec_parse': 'Parse',
						  'sec_create_x': 'Create X',
						  'sec_preprocessing': 'Preprocessing',
						  'sec_counter': 'Counter Offers',
						  #'sec_predict': 'Predict',
						  'sec_adv_act': 'Adverse Action',
						  'sec_gen_output': 'Generate Output'}
		# map
		df_for_plot['Steps'] = df_for_plot['Steps'].map(dict_map_steps)
		# title
		ax[4].set_title('Mean Seconds by Parsing Step by N Debtors')
		# create plot
		sns.barplot(x='Steps',
			        y='Mean Seconds',
			        hue='N Debtors',
			        data=df_for_plot,
			        ax=ax[4])
		# fix overlap
		plt.tight_layout()
		# save
		plt.savefig(str_filename, bbox_inches='tight')
		# save to object
		self.df_output_1 = df_output_1
		self.df_output_2 = df_output_2
		#self.fig = fig
		# return
		return self