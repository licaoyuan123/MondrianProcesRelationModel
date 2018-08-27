#from pylab import *
import csv
import numpy as np
import matplotlib
import os
matplotlib.use('Agg')
from matplotlib.pyplot import plot,savefig, close


print('Please input the folder Name: ')
folder  = raw_input()

file_list = os.listdir(folder)
result_plot_folder = folder+'plot'

if not os.path.exists(result_plot_folder):
	os.makedirs(result_plot_folder)
for file_name in file_list:
	with open(folder + os.sep+file_name) as f:
		reader = csv.reader(f)
		floatList = []
		for line in reader:
			plot(np.arange(len(line)),line)
			savefig(result_plot_folder+os.sep + file_name.split('.')[0] +'.png')
			close('all')

