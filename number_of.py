#------------------------------------------------Program_Details--------------------------------------------------------
# Created by: Rennspiess, E., Hoseini, M.
# Created on: March, 16, 2022
# Updated on: March, 16, 2022
# Compares observation data
#------------------------------------------------Libraries--------------------------------------------------------------
import numpy
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import optimize
import os
from sklearn.linear_model import LinearRegression
#------------------------------------------------Data-------------------------------------------------------------------
#SATELLITES:
reflector_all = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/results_final_all.csv')
reflector_s1c = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/results_final_s1c.csv')
reflector_s2x = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/results_final_s2x.csv')
reflector_s5x = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/results_final_s5x.csv')

#NUMBER OF SATELLITES:
number_of_sats = [reflector_s1c['no_of_sats'], reflector_s2x['no_of_sats'], reflector_s5x['no_of_sats']]
headers = ["S1C", "S2X", "S5X"]
number_of_satellites = pd.concat(number_of_sats, axis=1, keys=headers)

print(number_of_satellites["S1C"].sum(), number_of_satellites["S2X"].sum(), number_of_satellites["S5X"].sum())

#NUMBER OF GOOD OBSERVATIONS:
number_of_good = [reflector_s1c['no_of_good_obs'], reflector_s2x['no_of_good_obs'], reflector_s5x['no_of_good_obs']]
headers = ["S1C", "S2X", "S5X"]
number_of_good_observations = pd.concat(number_of_good, axis=1, keys=headers)

print(number_of_good_observations["S1C"].sum(), number_of_good_observations["S2X"].sum(),
      number_of_good_observations["S5X"].sum())

#NUMBER OF OUTLIERS:
