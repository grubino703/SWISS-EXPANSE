"""
Copyright 2022 Jan-Philipp Sasse (UNIGE), Evelina Trutnevyte (UNIGE)
~~~~~~~~~~~~~
Creates the directories to store the results

"""

# Import packages
import os
from EXPANSE.read_settings import *

# Read settings
opts = read_settings()

def create_directories():
	
	# Create output directory
	try:
	    os.mkdir(opts['output_path'])
	except OSError:
	    print("Creation of the directory %s failed" % opts['output_path'])
	else:
	    print("Successfully created the directory %s " % opts['output_path'])

	# Create summary directory
	sum_path = opts['output_path'] + 'Summary'

	try:
	    os.mkdir(sum_path)
	except OSError:
	    print("Creation of the directory %s failed" % sum_path)
	else:
	    print("Successfully created the directory %s " % sum_path)