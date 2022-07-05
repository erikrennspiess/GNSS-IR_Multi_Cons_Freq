#------------------------------------------------Program_Details--------------------------------------------------------
# Created by: Rennspiess, E., Hoseini, M.
# Created on: Feb, 16, 2022
# Updated on: May, 23, 2022
# Error analysis for comparing GNSS-R SNR data with Tide Gauge data from Onsala, Sweden.
#------------------------------------------------Libraries--------------------------------------------------------------
import numpy
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.polynomial.polynomial import polyfit
from scipy.signal import savgol_filter
from scipy import optimize
import os
import sklearn as sk
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

#------------------------------------------------Functions--------------------------------------------------------------
def clean_dataframe(frequency_dataframe):
    frequency_dataframe.loc[frequency_dataframe['no_of_outliers'] <= 1, ['estimated_reflector_height']] = np.nan
    frequency_dataframe.loc[frequency_dataframe['no_of_good_obs'] <= 3, ['estimated_reflector_height']] = np.nan
    obs_std_std = np.std(frequency_dataframe['estimated_reflector_height_std'])
    frequency_dataframe.loc[frequency_dataframe['estimated_reflector_height_std'] >= 2 * obs_std_std,
                            ['estimated_reflector_height_std']] = np.nan
    #obs_std = np.std(frequency_dataframe['estimated_reflector_height_std'])
    #frequency_dataframe.loc[frequency_dataframe['tg_sea_level'] >= 2 * obs_std,
                            #['estimated_reflector_height']] = np.nan
    return frequency_dataframe

def calculate_bias(tg_column, sea_level_column):
    diff = tg_column - sea_level_column
    bias = diff.mean()
    return bias

#Root Mean Squared Error (RMSE):
def rmse_function(tide_gauge, snr):
    MSE = np.square(np.subtract(tide_gauge, snr)).mean()
    RMSE = math.sqrt(MSE)
    return RMSE

#correlation function:
def correlation_function(sea_level_tg, reflector_height):
    return np.corrcoef(reflector_height, sea_level_tg)

#best fit line:
def best_fit_and_int(x, y):
    m = (((np.mean(x) * np.mean(y)) - np.mean(x*y)) /((np.mean(x) * np.mean(x)) - np.mean(x*x)))
    b = np.mean(y) - m*np.mean(x)
    return m, b

#regression line
def regression_line(x_freq, y_tg):
    rl_m, rl_b = best_fit_and_int(x_freq, y_tg)
    all_regression_line = [(rl_m * i) +
                           rl_b for i in x_freq]
    return all_regression_line

#------------------------------------------------Data-------------------------------------------------------------------
#SATELLITES & TIDEGAUGE DATA:
reflector_all = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/frequencies/'
                            'Galileo_frequencies/6_hour_window_galileo_all.csv')
reflector_s1c = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/frequencies/'
                            'Galileo_frequencies/6_hour_window_galileo_S1X.csv')
reflector_s5x = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/frequencies/'
                            'Galileo_frequencies/6_hour_window_galileo_S5X.csv')

#assign labels:
reflector_all['OB_Type'] = 'All'
reflector_s1c['OB_Type'] = 'S1C'
reflector_s5x['OB_Type'] = 'S5X'

#combine frames into 1:
frames = [reflector_all, reflector_s1c, reflector_s5x]
satellite_data_all_frequencies  = pd.concat(frames)

#convert time column to datetime
satellite_data_all_frequencies['time'] = pd.to_datetime(satellite_data_all_frequencies['time'])

#select date range:
start_date = '2018-09-01 00:00:00'
end_date = '2018-09-30 23:59:00'

#apply date masks to data frame:
date_mask = (satellite_data_all_frequencies['time'] > start_date) & (satellite_data_all_frequencies['time'] <= end_date)
satellite_data_all_frequencies = satellite_data_all_frequencies.loc[date_mask]

#clean data sets:
clean_dataframe(satellite_data_all_frequencies)

#create variables:
all_freq = satellite_data_all_frequencies.loc[satellite_data_all_frequencies['OB_Type'] == 'All']
s1c = satellite_data_all_frequencies.loc[satellite_data_all_frequencies['OB_Type'] == 'S1C']
s5x = satellite_data_all_frequencies.loc[satellite_data_all_frequencies['OB_Type'] == 'S5X']

#invert reflector heights + distance above water level:
all_freq['estimated_reflector_height'] = all_freq['estimated_reflector_height'].apply(lambda x: x * -1 + 3.00)
s1c['estimated_reflector_height'] = s1c['estimated_reflector_height'].apply(lambda x: x * -1.013 + 3.00)
s5x['estimated_reflector_height'] = s5x['estimated_reflector_height'].apply(lambda x: x * -1.017 + 3.00)


#------------------------------------------------RMSE Analysis----------------------------------------------------------
# calculate RMSE:
#calculate bias
bias_all = calculate_bias(all_freq['tg_sea_level'], all_freq['estimated_reflector_height'])
bias_S1C = calculate_bias(s1c['tg_sea_level'], s1c['estimated_reflector_height'])
bias_S5X = calculate_bias(s5x['tg_sea_level'], s5x['estimated_reflector_height'])

#apply rmse_function to dataframe:
#All frequencies->
rmse_all = rmse_function(all_freq['tg_sea_level'] - bias_all, all_freq['estimated_reflector_height'])
rmse_s1c = rmse_function(s1c['tg_sea_level'] - bias_S1C, s1c['estimated_reflector_height'])
rmse_s5x = rmse_function(s5x['tg_sea_level'] - bias_S5X, s5x['estimated_reflector_height'])

# correlation values
corr_all = all_freq['tg_sea_level'].corr(all_freq['estimated_reflector_height'])
corr_S1C = s1c['tg_sea_level'].corr(s1c['estimated_reflector_height'])
corr_S5X = s5x['tg_sea_level'].corr(s5x['estimated_reflector_height'])

print(rmse_all, rmse_s1c, rmse_s5x)
print(corr_all, corr_S1C, corr_S5X)
#------------------------------------------------Create Plots-----------------------------------------------------------
cmap = plt.cm.get_cmap('tab20c')
fig = plt.figure(figsize=(14, 5))
ax1 = fig.add_subplot(111)

#plot tide gauge:
ax1.plot(all_freq['time'], all_freq['tg_sea_level'])

#plot satellites:
ax1.plot(all_freq['time'], all_freq['estimated_reflector_height'])
#ax1.plot(all_freq['time'], all_freq['estimated_reflector_height'], color='r')
#ax1.plot(s1c['time'], s1c['estimated_reflector_height'], color=cmap(1))
#ax1.plot(s5x['time'], s5x['estimated_reflector_height'], color=cmap(9))

#ax2.set_xlabel("Date")
ax1.set_ylabel("Sea Level [m]", size=12)
ax1.set_xlabel("Date [yyyy-mm-dd]", size=12)
ax1.set_title('6 Hour Window All Frequencies vs Tide Gauge Measurements (Galileo)', size=14)

#plot legend
ax1.legend(['Tide Gauge', 'All Frequencies'], loc='upper left')
#ax1.legend(['Tide Gauge', 'All Frequencies', 'E1',  'E5a'], loc='upper left')

ax1.grid()
fig.tight_layout()


# Create second axes, the top-left plot with orange plot
fig2, ax23 = plt.subplots(1, 2, figsize=(14, 5))
sub2, sub3 = ax23.flatten()
x_axis = np.arange(0, 1)

sub2.bar(x_axis, rmse_all, color='r', width=1/4, label='Combined')
sub2.bar(x_axis + 1/3, rmse_s1c, color=cmap(1), width=1/4, label='S1C')
sub2.bar(x_axis + 2/3,  rmse_s5x, color=cmap(9), width=1/4, label='S5X')

#label x and y axis:
sub2.set_xlabel("Frequency")
sub2.set_ylabel("RMSE [m]")
sub2.set_title('RMSE Results')

#plot legend
x_ticks = [0, 1/3, 2/3]
x_labels = ['All Frequencies', 'E1', 'E5a']

#add x-axis values to plot
plt.xticks(ticks=x_ticks, labels=x_labels)
sub2.set_xticks(ticks=x_ticks, labels=x_labels)

# Create third axes, a combination of third and fourth cell
sub3.bar(x_axis, np.mean(s1c['no_of_good_obs'].dropna().values + s5x['no_of_good_obs'].dropna().values), color='r', width=1/4, label='Combined')
sub3.bar(x_axis + 1/3, np.mean(s1c['no_of_good_obs'].dropna().values), color=cmap(1), width=1/4, label='E1')
sub3.bar(x_axis + 2/3, np.mean(s5x['no_of_good_obs'].dropna().values), color=cmap(9), width=1/4, label='E5a')

#label x and y axis:
sub3.set_xlabel("Frequency")
sub3.set_ylabel("Number of Observations")
sub3.set_title('Observations Results')

#plot legend
x_ticks = [0, 1/3, 2/3]
x_labels = ['All Frequencies', 'E1', 'E5a']

#add x-axis values to plot
plt.xticks(ticks=x_ticks, labels=x_labels)
fig2.tight_layout()


#create regression lines:
#All freq
all_regression_line = regression_line(all_freq['estimated_reflector_height'], all_freq['tg_sea_level'])
s1c_regression_line = regression_line(s1c['estimated_reflector_height'], s1c['tg_sea_level'])
s5x_regression_line = regression_line(s5x['estimated_reflector_height'], s5x['tg_sea_level'])

# make subplots
fig3 = plt.figure()

ax3 = plt.subplot(211)
ax4 = plt.subplot(223)
ax5 = plt.subplot(224)

norm = plt.Normalize(vmin=0, vmax=2)
cmap = plt.cm.get_cmap('RdYlBu_r')
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax3, label='Estimated Reflector Height STD')

ax3.scatter(all_freq['estimated_reflector_height']+0.2, all_freq['tg_sea_level'],
                 c=all_freq['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'), s=0.9)
ax4.scatter(s1c['estimated_reflector_height']+0.2, s1c['tg_sea_level'],
                 c=s1c['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'), s=0.9)
ax5.scatter(s5x['estimated_reflector_height']+0.2, s5x['tg_sea_level'],
                 c=s5x['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'), s=0.9)

ax3.plot(all_freq['estimated_reflector_height']+0.2, all_regression_line, color='k')
ax4.plot(s1c['estimated_reflector_height']+0.2, s1c_regression_line, color='k')
ax5.plot(s5x['estimated_reflector_height']+0.2, s5x_regression_line, color='k')

plt.suptitle('3 Hour Window Correlation Values: Tide Gauge vs Reflector Height')
ax3.set_title("All Frequencies")
ax4.set_title("E1")
ax5.set_title("E5a")

ax3.set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")
ax4.set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")
ax5.set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")

ax3.grid()
ax4.grid()
ax5.grid()

fig3.tight_layout()

#fig.savefig('/home/OceanJasper/GNSS-R/altimetry/final_thesis_figures/galileo/6_hour_timeseries_multi')
fig2.savefig('/home/OceanJasper/GNSS-R/altimetry/final_thesis_figures/galileo/6_hour_rmse_epoch')
#fig3.savefig('/home/OceanJasper/GNSS-R/altimetry/final_thesis_figures/galileo/3_hour_correlation_plot_with_regression')


plt.show()




