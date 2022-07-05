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
reflector_all = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/frequencies/GPS_frequencies/'
                            '6hour_window_all_azi.csv')
reflector_s1c = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/frequencies/GPS_frequencies/'
                            '6hour_window_S1C_azi.csv')
reflector_s2x = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/frequencies/GPS_frequencies/'
                            '6hour_window_s2x_azi.csv')
reflector_s5x = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/frequencies/GPS_frequencies/'
                            '6hour_window_s5x_azi.csv')

#assign labels:
reflector_all['OB_Type'] = 'All'
reflector_s1c['OB_Type'] = 'S1C'
reflector_s2x['OB_Type'] = 'S2X'
reflector_s5x['OB_Type'] = 'S5X'

#combine frames into 1:
frames = [reflector_all, reflector_s1c, reflector_s2x, reflector_s5x]
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
s2x = satellite_data_all_frequencies.loc[satellite_data_all_frequencies['OB_Type'] == 'S2X']
s5x = satellite_data_all_frequencies.loc[satellite_data_all_frequencies['OB_Type'] == 'S5X']

#invert reflector heights + distance above water level:
all_freq['estimated_reflector_height'] = all_freq['estimated_reflector_height'].apply(lambda x: x * -1 + 3.00)
s1c['estimated_reflector_height'] = s1c['estimated_reflector_height'].apply(lambda x: x * -1.013 + 3.00)
s2x['estimated_reflector_height'] = s2x['estimated_reflector_height'].apply(lambda x: x * -1.015 + 3.00)
s5x['estimated_reflector_height'] = s5x['estimated_reflector_height'].apply(lambda x: x * -1.017 + 3.00)


#------------------------------------------------RMSE Analysis----------------------------------------------------------
# calculate RMSE:
#calculate bias
bias_all = calculate_bias(all_freq['tg_sea_level'], all_freq['estimated_reflector_height'])
bias_S1C = calculate_bias(s1c['tg_sea_level'], s1c['estimated_reflector_height'])
bias_S2X = calculate_bias(s2x['tg_sea_level'], s2x['estimated_reflector_height'])
bias_S5X = calculate_bias(s5x['tg_sea_level'], s5x['estimated_reflector_height'])

#apply rmse_function to dataframe:
#All frequencies->
rmse_all = rmse_function(all_freq['tg_sea_level'] - bias_all, all_freq['estimated_reflector_height'])
rmse_s1c = rmse_function(s1c['tg_sea_level'] - bias_S1C, s1c['estimated_reflector_height'])
rmse_s2x = rmse_function(s2x['tg_sea_level'] - bias_S2X, s2x['estimated_reflector_height'])
rmse_s5x = rmse_function(s5x['tg_sea_level'] - bias_S5X, s5x['estimated_reflector_height'])

# correlation values
corr_all = all_freq['tg_sea_level'].corr(all_freq['estimated_reflector_height'])
corr_S1C = s1c['tg_sea_level'].corr(s1c['estimated_reflector_height'])
corr_S2X = s2x['tg_sea_level'].corr(s2x['estimated_reflector_height'])
corr_S5X = s5x['tg_sea_level'].corr(s5x['estimated_reflector_height'])

print(rmse_all, rmse_s1c, rmse_s2x, rmse_s5x)
print(corr_all, corr_S1C, corr_S2X, corr_S5X)

#------------------------------------------------Create Plots-----------------------------------------------------------
for col in satellite_data_all_frequencies.columns:
    print(col)

cmap = plt.cm.get_cmap('tab20c')
fig = plt.figure(figsize = (14,5))
ax1 = fig.add_subplot(111)

#plot tide gauge:
ax1.plot(all_freq['time'], all_freq['tg_sea_level'])

#plot satellites:
ax1.plot(all_freq['time'], all_freq['estimated_reflector_height'])
ax1.plot(s1c['time'], s1c['estimated_reflector_height'])
ax1.plot(s2x['time'], s2x['estimated_reflector_height'])
ax1.plot(s5x['time'], s5x['estimated_reflector_height'])
#ax2.set_xlabel("Date")
ax1.set_ylabel("Sea Level [m]", size=12)
ax1.set_xlabel("Date [yyyy-mm-dd]", size=12)
ax1.set_title('6 Hour Window All Frequencies vs Tide Gauge Measurements', size=12)
#plot legend
#ax1.legend(['Tide Gauge', 'Multiple Frequencies'], loc='upper left')
ax1.legend(['Tide Gauge', 'All Frequencies', 'S1C', 'S2X', 'S5X'], loc='upper left')
ax1.grid()
fig.tight_layout()



# Create second axes, the top-left plot with orange plot
fig2, ax23 = plt.subplots(1, 2, figsize=(14, 5))
sub2, sub3 = ax23.flatten()

x_axis = np.arange(0, 1)

sub2.bar(x_axis, rmse_all, color=cmap(0), width=1/5, label='Combined')
sub2.bar(x_axis + 1/4, rmse_s1c, color=cmap(4), width=1/5, label='S1C')
sub2.bar(x_axis + 2/4, rmse_s2x, color=cmap(5), width=1/5, label='S2X')
sub2.bar(x_axis + 3/4,  rmse_s5x, color=cmap(8), width=1/5, label='S5X')

#label x and y axis:
sub2.set_xlabel("Frequency")
sub2.set_ylabel("RMSE [m]")
sub2.set_title('RMSE Results')

#plot legend
#ax3.legend(['S125', 'S1C', 'S2X', 'S5X'])
x_ticks = [0.0, 0.25, 0.5, 0.75]
x_labels = ['All Frequencies', 'S1C', 'S2X', 'S5X']

#add x-axis values to plot
plt.xticks(ticks=x_ticks, labels=x_labels)
sub2.set_xticks(ticks=x_ticks, labels=x_labels)

# Create third axes, a combination of third and fourth cell
#option 1:
sub3.bar(x_axis, np.mean(all_freq['no_of_good_obs'].dropna().values), color=cmap(0), width=1/5, label='Combined')
sub3.bar(x_axis + 1/4, np.mean(s1c['no_of_good_obs'].dropna().values), color=cmap(4), width=1/5, label='S1C')
sub3.bar(x_axis + 2/4, np.mean(s2x['no_of_good_obs'].dropna().values), color=cmap(5), width=1/5, label='S2X')
sub3.bar(x_axis + 3/4, np.mean(s5x['no_of_good_obs'].dropna().values), color=cmap(8), width=1/5, label='S5X')

#label x and y axis:
sub3.set_xlabel("Frequency")
sub3.set_ylabel("Number of Observations")
sub3.set_title('Observations Results')
#plot legend
#option 1:
x_ticks = [0.0, 0.25, 0.5, 0.75]
x_labels = ['All Frequencies', 'S1C', 'S2X', 'S5X']

#add x-axis values to plot
plt.xticks(ticks=x_ticks, labels=x_labels)



# make subplots

all_regression_line = regression_line(all_freq['estimated_reflector_height'], all_freq['tg_sea_level'])
s1c_regression_line = regression_line(s1c['estimated_reflector_height'], s1c['tg_sea_level'])
s2x_regression_line = regression_line(s2x['estimated_reflector_height'], s2x['tg_sea_level'])
s5x_regression_line = regression_line(s5x['estimated_reflector_height'], s5x['tg_sea_level'])

fig3, ax = plt.subplots(2, 2)

# set data with subplots and plot
ax[0, 0].scatter(all_freq['estimated_reflector_height'], all_freq['tg_sea_level'],
                 c=all_freq['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'), s=0.9)

ax[0, 1].scatter(s1c['estimated_reflector_height'], s1c['tg_sea_level'],
                 c=s1c['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'), s=0.9)

ax[1, 0].scatter(s2x['estimated_reflector_height'], s2x['tg_sea_level'],
                 c=s2x['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'), s=0.9)

ax[1, 1].scatter(s5x['estimated_reflector_height'], s5x['tg_sea_level'],
                 c=s5x['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'), s=0.9)

#plot regression line:
ax[0, 0].plot(all_freq['estimated_reflector_height'], all_regression_line, color='k',alpha=0.8)
ax[0, 1].plot(s1c['estimated_reflector_height'], s1c_regression_line, color='k')
ax[1, 0].plot(s2x['estimated_reflector_height'], s2x_regression_line, color='k')
ax[1, 1].plot(s5x['estimated_reflector_height'], s5x_regression_line, color='k')

ax[0,0].grid()
ax[0,1].grid()
ax[1,0].grid()
ax[1,1].grid()

# set the title to subplots
ax[0, 0].set_title("All Frequencies")
ax[0, 1].set_title("S1C")
ax[1, 0].set_title("S2X")
ax[1, 1].set_title("S5X")

# set labels
#plt.setp(ax[-1, :], xlabel='Tide Gauge Sea Level')
#plt.setp(ax[:, 0], ylabel='Estimated Reflector Height')

ax[0, 0].set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")
ax[0, 1].set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")
ax[1, 0].set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")
ax[1, 1].set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")
plt.suptitle('3 Hour Window Correlation Values: Tide Gauge vs Reflector Height')

# set spacing
fig3.tight_layout()
norm = mpl.colors.Normalize(vmin=0, vmax=2)
fig3.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.get_cmap('RdYlBu_r')), ax=ax.ravel(), orientation='vertical',
                                   label='Estimated Reflector Height STD')

fig.savefig('/home/OceanJasper/GNSS-R/altimetry/final_thesis_figures/gps/6_hour_timeseries_ind')
#fig2.savefig('/home/OceanJasper/GNSS-R/altimetry/final_thesis_figures/gps/6_hour_rmse_epoch')
#fig3.savefig('/home/OceanJasper/GNSS-R/altimetry/final_thesis_figures/gps/3_hour_correlation_plot_with_reggression')
plt.show()

