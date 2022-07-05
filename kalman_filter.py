#------------------------------------------------Program_Details--------------------------------------------------------
# Created by: Rennspiess, E., Hoseini, M.
# Kalman filter algorithm by: Andrés Echeverri Created on Mon Oct 28 20:16:36 2019
# Created on: Feb, 16, 2022
# Updated on: March, 01, 2022
# Error analysis for comparing GNSS-R SNR data with Tide Gauge data from Onsala, Sweden.
#------------------------------------------------Libraries--------------------------------------------------------------
import matplotlib.dates as md
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import inv
#------------------------------------------------Functions--------------------------------------------------------------
#Root Mean Squared Error (RMSE):
def rmse_function(snr, tide_gauge):
    MSE = np.square(np.subtract(tide_gauge, snr)).mean()
    RMSE = math.sqrt(MSE)
    return RMSE

def clean_dataframe_I(frequency_dataframe):
    frequency_dataframe.loc[frequency_dataframe['no_of_outliers'] <= 1, ['estimated_reflector_height']] = np.nan
    frequency_dataframe.loc[frequency_dataframe['no_of_good_obs'] <= 3, ['estimated_reflector_height']] = np.nan
    return frequency_dataframe

def clean_dataframe_II(frequency_dataframe):
    frequency_dataframe.loc[frequency_dataframe['no_of_outliers'] <= 1, ['estimated_reflector_height']] = np.nan
    frequency_dataframe.loc[frequency_dataframe['no_of_good_obs'] <= 3, ['estimated_reflector_height']] = np.nan
    obs_std = np.std(frequency_dataframe['estimated_reflector_height_std'])
    frequency_dataframe.loc[frequency_dataframe['estimated_reflector_height_std'] >= 1 * obs_std,
                            ['estimated_reflector_height']] = np.nan
    obs_tg_std = np.std(frequency_dataframe['tg_sea_level'])
    frequency_dataframe.loc[frequency_dataframe['estimated_reflector_height_std'] >= 1 * obs_tg_std,
                            ['estimated_reflector_height']] = np.nan
    return frequency_dataframe

#------------------------------------------------Data-------------------------------------------------------------------
reflector_all = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/frequencies/6hour_window_all.csv')

clean_dataframe_II(reflector_all)

tide_gauge = [reflector_all['time'], reflector_all['tg_sea_level'],
              reflector_all['estimated_reflector_height'] * -1 + 3.00, reflector_all['no_of_good_obs']]
headers = ["Time", "TG", "All", "Count"]

tg_values = pd.concat(tide_gauge, axis=1, keys=headers)

y = tg_values['All']


"""
Finding some statistics of signal is always a good practice, the most used ones
for our purpose are the mean nd the standard deviation. Howerver, it is necessary 
to find an interval where the signal does not have oscilations.
"""
datenums = md.date2num(tg_values['Time'])
xfmt = md.DateFormatter('%Y-%m-%d')

signal_noise = y
mean = np.mean(signal_noise)
std = np.std(signal_noise)
mean_line = np.ones(len(signal_noise), dtype=int) * mean
std_line_1 = np.ones(len(signal_noise), dtype=int) * (mean + std)
std_line_2 = np.ones(len(signal_noise), dtype=int) * (mean - std)

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.xaxis.set_major_formatter(xfmt)
ax1.plot(datenums, signal_noise, label="Signal")
ax1.plot(datenums, mean_line, label="mean")
ax1.plot(datenums, std_line_1, color='red', label="std", linestyle='--')
ax1.plot(datenums, std_line_2, color='red',  linestyle='--')
ax1.set_xlabel("Date")
ax1.set_ylabel("Sea Level [m]")
ax1.legend()

# Transition Matrix
A = np.array([[1.0]])
# Observation Matrix
C = np.array([[1.0]])
# Process Noise Covariance
Rww = np.array([[1]])
# Measurement Noise Covariance
Rvv = np.array([[1 * 1]])
# Control action Matrix
B = np.array([0])
# Control input
U = np.array([0])
# state vector
x = np.zeros((len(y), 1))
# Covariance Matrix
P = np.zeros((len(y), 1))
# Weighted MD vector
MDw = np.zeros((len(y), 1))
# Initial Covariance Value
P[0] = 0
I = np.identity(1)

"""
Kalman filter implementation: Most implementations can be divided in 2 or 3
steps. Some consider innovation as part of the prediction. But, for this 
implementation it is nice to see where the Mahalanobis distance (MD) is taking 
place within the measurement noise covariance calculation. The approximated MD
is used due to its simplicity. 
"""
for i in range(1, len(y)):
    # Initialization of the vector state
    x[0] = y[0]
    """
    Prediction
    """
    x[i] = dot(A, x[i - 1]) + dot(B, U)
    P[i] = dot(A, dot(P[i], A.T)) + Rww

    """
    Innovation
    """
    e = y[i] - C * x[i]
    Ree = dot(C, dot(P[i], C.T)) + Rvv
    # Mahalanobbis distance approximation
    MD = math.sqrt(e * e) / Ree
    # Weighted MD
    MDw[i] = 1 / (1 + (math.exp(-MD) + 0.1))
    # New Measurement Noise Covariance
    Rvv = np.array([[4 * MDw[i]]])
    # Kalman gain
    K = dot(P[i], dot(C.T, inv(Ree)))
    """
    Update
    """
    x[i] = x[i] + dot(K, dot(e, K))
    P[i] = dot(I - dot(K, C), P[i])

size = tg_values['Count'].to_numpy()
s = [s*.2 for s in size]

fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.xaxis.set_major_formatter(xfmt)
ax2.plot(datenums, tg_values['TG'], label="Tide Gauge")
ax2.plot(datenums, y, label="Filtered Signal")
#ax2.scatter(datenums, x, label="Filtered Signal")
ax2.set_xlabel("Date")
ax2.set_ylabel("Sea Level [m]")
ax2.legend()

#------------------------------------------------Error_Analysis---------------------------------------------------------
# build dataframe for error analysis:
unfiltered_df = pd.DataFrame(y)
filtered_df = pd.DataFrame(x)
rmse_tg_all = [tg_values['Time'], tg_values['TG'], unfiltered_df, filtered_df]
rmse_tg = pd.concat(rmse_tg_all, axis=1)

#rename columns:
rmse_tg.rename(columns = {'All':'Unfiltered', 0:'Filtered'}, inplace = True)

#apply rmse_function to dataframe:
rmse_all_unfiltered_filtered = []
for date, i in rmse_tg.groupby('Time'):
    rmse_unfiltered = rmse_function(i['Unfiltered'], (i['TG']))
    rmse_filtered = rmse_function(i['Filtered'], (i['TG']))
    rmse_all_unfiltered_filtered.append([date, rmse_unfiltered, rmse_filtered])

rmse_results = pd.DataFrame(rmse_all_unfiltered_filtered, columns=['Date', 'Unfiltered', 'Filtered'])
rmse_results['Date'] = pd.to_datetime(rmse_results['Date'])
rmse_results = rmse_results.resample('D', on='Date').mean()
rmse_results = rmse_results.reset_index(drop=False)

plt.show()

fig3 = plt.figure(111)
ax3 = fig3.add_subplot(111)
x_axis = np.arange(len(rmse_results['Date']))
ax3.bar(x_axis -0.3, rmse_results['Unfiltered'], width=0.3, label='Unfiltered')
ax3.bar(x_axis +0.3, rmse_results['Filtered'], width=0.3, label='Filtered')

#label x and y axis:
ax3.set_xlabel("Date")
ax3.set_ylabel("Sea Level [m]")
#plot legend
ax3.legend(['UnFiltered', 'Filtered'])
