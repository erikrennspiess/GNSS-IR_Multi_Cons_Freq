#------------------------------------------------Program_Details--------------------------------------------------------
# Created by: Rennspiess, E., Hoseini, M.
# Created on: Feb, 16, 2022
# Updated on: May, 23, 2022
# Error analysis for comparing GNSS-R SNR data with Tide Gauge data from Onsala, Sweden.
#------------------------------------------------Libraries--------------------------------------------------------------
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

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
                            'Glonass_frequencies/3_hour_window_glonass_all.csv')
reflector_s1c = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/frequencies/'
                            'Glonass_frequencies/3_hour_window_glonass_S1C.csv')
reflector_s2c = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/frequencies/'
                            'Glonass_frequencies/3_hour_window_glonass_S2C.csv')

#TIDEGAUGE:
tidegauge = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/tide_gauge_data/tg_data.txt', sep=';')
tidegauge = tidegauge.drop(tidegauge.columns[[2, 3, 4, 5]], axis=1)

#convert time column to datetime:
tidegauge['time'] = pd.to_datetime(tidegauge['time'])

#convert from cm to m:
tidegauge['sea_level'] = tidegauge['sea_level'] * .01

#select date range:
start_date = '2018-09-01 00:00:00'
end_date = '2018-09-30 23:59:00'

#apply date masks to data frame:
date_mask = (tidegauge['time'] > start_date) & (tidegauge['time'] <= end_date)
tide_gauge_date_selection = tidegauge.loc[date_mask]

#assign labels:
reflector_all['OB_Type'] = 'All'
reflector_s1c['OB_Type'] = 'S1C'
reflector_s2c['OB_Type'] = 'S2C'

#combine frames into 1:
frames = [reflector_all, reflector_s1c, reflector_s2c]
satellite_data_all_frequencies = pd.concat(frames)

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
s2c = satellite_data_all_frequencies.loc[satellite_data_all_frequencies['OB_Type'] == 'S2C']

#invert reflector heights + distance above water level:
all_freq['estimated_reflector_height'] = all_freq['estimated_reflector_height'].apply(lambda x: x * -1 + 3.00)
s1c['estimated_reflector_height'] = s1c['estimated_reflector_height'].apply(lambda x: x * -1.013 + 3.00)
s2c['estimated_reflector_height'] = s2c['estimated_reflector_height'].apply(lambda x: x * -1.017 + 3.00)


#------------------------------------------------RMSE Analysis----------------------------------------------------------
# calculate RMSE:
#calculate bias
bias_all = calculate_bias(all_freq['tg_sea_level'], all_freq['estimated_reflector_height'])
bias_S1C = calculate_bias(s1c['tg_sea_level'], s1c['estimated_reflector_height'])
bias_S2C = calculate_bias(s2c['tg_sea_level'], s2c['estimated_reflector_height'])

#apply rmse_function to dataframe:
#All frequencies->
rmse_all = rmse_function(all_freq['tg_sea_level'] - bias_all, all_freq['estimated_reflector_height'])
rmse_s1c = rmse_function(s1c['tg_sea_level'] - bias_S1C, s1c['estimated_reflector_height'])
rmse_s2c = rmse_function(s2c['tg_sea_level'] - bias_S2C, s2c['estimated_reflector_height'])

# correlation values
corr_all = all_freq['tg_sea_level'].corr(all_freq['estimated_reflector_height'])
corr_S1C = s1c['tg_sea_level'].corr(s1c['estimated_reflector_height'])
corr_S2C = s2c['tg_sea_level'].corr(s2c['estimated_reflector_height'])

print(rmse_all, rmse_s1c, rmse_s2c)
print(corr_all, corr_S1C, corr_S2C)
#------------------------------------------------Create Plots-----------------------------------------------------------
for col in satellite_data_all_frequencies.columns:
    print(col)

cmap = plt.cm.get_cmap('tab20c')
fig = plt.figure(figsize=(14, 5))
ax1 = fig.add_subplot(111)
#plot tide gauge:
sns.lineplot(tide_gauge_date_selection['time'], tide_gauge_date_selection['sea_level'], ax=ax1)
#plot satellites:
#sns.lineplot(x=all_freq['time'], y=all_freq['estimated_reflector_height'], marker='o', linestyle='-', ax=ax1)
sns.lineplot(x=all_freq['time'], y=all_freq['estimated_reflector_height'], marker='o', linestyle='-', color='r', ax=ax1)
sns.lineplot(s1c['time'], s1c['estimated_reflector_height'], marker='o', linestyle='-', color=cmap(1), ax=ax1)
sns.lineplot(s2c['time'], s2c['estimated_reflector_height'], marker='o', linestyle='-', color=cmap(9), ax=ax1)

#ax2.set_xlabel("Date")
ax1.set_ylabel("Sea Level [m]", size=12)
ax1.set_xlabel("Date [yyyy-mm-dd]", size=12)
ax1.set_title('6 Hour Window All Frequencies vs Tide Gauge Measurements (GLONASS)', size=14)

#plot legend
ax1.legend(labels=['Tide Gauge', 'All Frequencies', 'S1C',  'S2C'])
ax1.grid()


fig.tight_layout()
plt.show()

# Create second axes, the top-left plot with orange plot
fig2, ax23 = plt.subplots(1, 2, figsize=(14, 5))
sub2, sub3 = ax23.flatten()

x_axis = np.arange(0, 1)
sub2.bar(x_axis, rmse_all, color='r', width=1/4, label='Combined')
sub2.bar(x_axis + 1/3, rmse_s1c, color=cmap(1), width=1/4, label='S1C')
sub2.bar(x_axis + 2/3,  rmse_s2c, color=cmap(9), width=1/4, label='S2C')

#label x and y axis:
sub2.set_xlabel("Frequency")
sub2.set_ylabel("RMSE [m]")
sub2.set_title('RMSE Results')
#plot legend
x_ticks = [0, 1/3, 2/3]
x_labels = ['All Frequencies', 'S1C', 'S2C']

#add x-axis values to plotglonass_reflector_tg_analysis
sub2.set_xticks(ticks=x_ticks, labels=x_labels)

# Create third axes, a combination of third and fourth cell
#option 1:
sub3.bar(x_axis, np.mean(all_freq['no_of_good_obs'].dropna().values), color='r', width=1/5, label='Combined')
sub3.bar(x_axis + 1/3, np.mean(s1c['no_of_good_obs'].dropna().values), color=cmap(1), width=1/5, label='S1C')
sub3.bar(x_axis + 2/3, np.mean(s2c['no_of_good_obs'].dropna().values), color=cmap(9), width=1/5, label='S2C')

#label x and y axis:
sub3.set_xlabel("Frequency")
sub3.set_ylabel("Number of Observations")
sub3.set_title('Observations Results')
#plot legend
#option 1:
x_ticks = [0, 1/3, 2/3]
x_labels = ['S12', 'S1C', 'S2C']

#add x-axis values to plot
plt.xticks(ticks=x_ticks, labels=x_labels)

#create regression lines:
#All freq
all_regression_line = regression_line(all_freq['estimated_reflector_height'], all_freq['tg_sea_level'])
s1c_regression_line = regression_line(s1c['estimated_reflector_height'], s1c['tg_sea_level'])
s2c_regression_line = regression_line(s2c['estimated_reflector_height'], s2c['tg_sea_level'])

# make subplots
fig3 = plt.figure()

ax3 = plt.subplot(211)
ax4 = plt.subplot(223)
ax5 = plt.subplot(224)

all_sctr = ax3.scatter(all_freq['estimated_reflector_height'], all_freq['tg_sea_level'],
                 c=all_freq['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'), s=0.9) #
plt.colorbar(all_sctr, ax=ax3, label='Estimated Reflector Height STD')

ax4.scatter(s1c['estimated_reflector_height'], s1c['tg_sea_level'],
                 c=s1c['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'), s=0.9)
ax5.scatter(s2c['estimated_reflector_height'], s2c['tg_sea_level'],
                 c=s2c['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'), s=0.9)


ax3.plot(all_freq['estimated_reflector_height'], all_regression_line, color='k')
ax4.plot(s1c['estimated_reflector_height'], s1c_regression_line, color='k')
ax5.plot(s2c['estimated_reflector_height'], s2c_regression_line, color='k')

plt.suptitle('3 Hour Window Correlation Values: Tide Gauge vs Reflector Height')
ax3.set_title("All Frequencies")
ax4.set_title("GLO-S1C")
ax5.set_title("GLO-S2C")

ax3.set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")
ax4.set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")
ax5.set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")

ax3.grid()
ax4.grid()
ax5.grid()

fig3.tight_layout()

#fig.savefig('/home/OceanJasper/GNSS-R/altimetry/final_thesis_figures/glonass/3_hour_timeseries_all')
#fig2.savefig('/home/OceanJasper/GNSS-R/altimetry/final_thesis_figures/glonass/3_hour_rmse_epoch')
fig3.savefig('/home/OceanJasper/GNSS-R/altimetry/final_thesis_figures/glonass/3_hour_correlation_plot_with_regression')

plt.show()

