# preprocessing
from .leaky_features_4 import LeakyFeatures

# define class
class CreateConstants(LeakyFeatures):
	# initialize
	def __init__(self, str_dirname_output='./opt/ml/model', str_target='target', str_datecol='date', bool_target_binary=True):
		# initialize parent class
		LeakyFeatures.__init__(self, str_dirname_output, str_target, str_datecol, bool_target_binary)
		# save arguments to object
		self.str_dirname_output = str_dirname_output
		self.str_target = str_target
		self.str_datecol = str_datecol
		self.bool_target_binary = bool_target_binary
	# map term
	def custom_mapping_term(self, int_term):
		if int_term == 0:
			return 72
		elif int_term <= 12:
			return 12
		elif int_term <= 24:
			return 24
		elif int_term <= 36:
			return 36
		elif int_term <= 48:
			return 48
		elif int_term <= 60:
			return 60
		else:
			return 72
	# map PTI
	def custom_mapping_pti(self, flt_pti):
		if flt_pti <= 0:
			return 0.15
		elif flt_pti <= 0.03:
			return 0
		elif flt_pti <= 0.06:
			return 0.03
		elif flt_pti <= 0.09:
			return 0.06
		elif flt_pti <= 0.12:
			return 0.09
		elif flt_pti <= 0.15:
			return 0.12
		else:
			return 0.15

'''
to use previous class
1. call the current class
	cls_current = current_module(arguments=arguments)

2. call previous module from current class, assign to new object perhaps with previous module name
	cls_previous = cls_current.previous_module(argument=argument...)

3. make sure to review arguments from previous module in order to update:
	file name, path, etc. 


'''