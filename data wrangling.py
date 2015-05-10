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

plt.hist(day.cnt, bins=100, color='#cccccc')
plt.xlabel("Total count of users")

# Overlay the histogram of casual users and registered users on the same graph
# casual user = blue; registered user = red, 50% transparency 
plt.hist(day.casual, bins=50, normed=True, color="#6495ED", alpha=.5) 
plt.hist(day.registered, bins=50, normed=True, color="#F08080", alpha=.5);
#--> More registered users use the bike than casual users

