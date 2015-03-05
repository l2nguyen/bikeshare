#------------------------------------------------------------------#
# Homework: Looking at the Capital Bikeshare data
#
# Note: Data downloaded from kaggle
#------------------------------------------------------------------#
# Last edited by: Lena Nguyen, March 5, 2015 
#------------------------------------------------------------------#

#----------------#
# IMPORT MODULES #
#----------------#

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#-------------#
# IMPORT DATA #
#-------------#

# Read drinks.csv into a DataFrame called 'drinks'
day = pd.read_csv('../data/day.csv')

# look at the first 10 line just to see the data
day.head(10)

day.describe()

plt.hist(day.registered, bins=100, color='#cccccc')
plt.xlabel("Registered")