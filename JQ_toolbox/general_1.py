import os
import pickle
import pandas as pd

# define new class to be inherited in the next module

class General:
	# initialize
	def __init__(self, str_dirname_output='./output'):
		self.str_dirname_output = str_dirname_output
	# create directory
	def create_directory(self):
		# create output dir
		try:
			os.mkdir(f'{self.str_dirname_output}')
			print(f'Created directory {self.str_dirname_output}')
		except FileExistsError:
			print(f'Directory {self.str_dirname_output} already exists')
		# return object
		return self
	# pickle to file
	def pickle_to_file(self, item_to_pickle, str_filename='cls_eda.pkl'):
		# save
		pickle.dump(item_to_pickle, open(f'{self.str_dirname_output}/{str_filename}', 'wb'))
		# return object
		return self

'''
cls_gen = General(str_dirname_output='./opt/ml/model',)

cls_gen.create_directory()

cls_gen.pickle_to_file(
	item_to_pickle=cls_general,
	str_filename='cls_general.pkl',
	)
'''