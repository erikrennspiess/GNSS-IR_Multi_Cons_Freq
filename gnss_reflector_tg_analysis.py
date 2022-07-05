#------------------------------------------------Program_Details--------------------------------------------------------
# Created by: Rennspiess, E., Hoseini, M.
# Created on: Feb, 16, 2022
# Updated on: May, 20, 2022
# Error analysis for comparing GNSS-R SNR data with Tide Gauge data from Onsala, Sweden.
#------------------------------------------------Libraries--------------------------------------------------------------
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl


#------------------------------------------------Functions--------------------------------------------------------------
def clean_dataframe(frequency_dataframe):
    frequency_dataframe.loc[frequency_dataframe['no_of_outliers'] <= 1, ['estimated_reflector_height']] = np.nan
    frequency_dataframe.loc[frequency_dataframe['no_of_good_obs'] <= 3, ['estimated_reflector_height']] = np.nan
    obs_std_std = np.std(frequency_dataframe['estimated_reflector_height_std'])
    frequency_dataframe.loc[frequency_dataframe['estimated_reflector_height_std'] >= 2 * obs_std_std,
                            ['estimated_reflector_height_std']] = np.nan
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

#------------------------------------------------Import Data------------------------------------------------------------
#SATELLITES & TIDEGAUGE DATA:
frequencies_all = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/frequencies/'
                            'GNSS_frequencies/6_hour_window_gps_galielo_glo_all.csv')
#L1 Band:
gps_s1c = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/frequencies/'
                            'GPS_frequencies/6hour_window_S1C_azi.csv')

galileo_e1 = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/frequencies/Galileo_frequencies/'
                         '6_hour_window_galileo_S1X.csv')
glonass_s1c = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/frequencies/Glonass_frequencies/'
                          '6_hour_window_glonass_S1C.csv')

#L2 Band:
gps_s2x = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/frequencies/GPS_frequencies/'
                      '6hour_window_s2x_azi.csv')
glonass_s2c = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/frequencies/Glonass_frequencies/'
                          '6_hour_window_glonass_S2C.csv')

#L5 Band:
gps_s5x = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/frequencies/GPS_frequencies/'
                      '6hour_window_S5X.csv')
galileo_e5 = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/frequencies/Galileo_frequencies/'
                         '6_hour_window_galileo_S5X.csv')

#------------------------------------------------Process Data-----------------------------------------------------------
#assign labels:
frequencies_all['OB_Type'] = 'All'
gps_s1c['OB_Type'] = 'GPS-S1C'
galileo_e1['OB_Type'] = 'GAL-E1'
glonass_s1c['OB_Type'] = 'GLO-S1C'
gps_s2x['OB_Type'] = 'GPS-S2X'
glonass_s2c['OB_Type'] = 'GLO-S2C'
gps_s5x['OB_Type'] = 'GPS-S5X'
galileo_e5['OB_Type'] = 'GAL-E5a'


#combine frames into 1:
frames = [frequencies_all, gps_s1c, galileo_e1, glonass_s1c, gps_s2x, glonass_s2c, gps_s5x, galileo_e5]
satellite_data_all_frequencies = pd.concat(frames)

#convert time column to datetime
satellite_data_all_frequencies['time'] = pd.to_datetime(satellite_data_all_frequencies['time'])

#select date range:
start_date = '2018-09-01 00:00:00'
end_date = '2018-09-30 23:59:00'

#apply date masks to data frame:
date_mask = (satellite_data_all_frequencies['time'] > start_date) & (satellite_data_all_frequencies['time'] <= end_date)
satellite_data_all_frequencies = satellite_data_all_frequencies.loc[date_mask]

#clean data set:
clean_dataframe(satellite_data_all_frequencies)

#create variables:
all_freq = satellite_data_all_frequencies.loc[satellite_data_all_frequencies['OB_Type'] == 'All']

#L1 Band:
s1c = satellite_data_all_frequencies.loc[satellite_data_all_frequencies['OB_Type'] == 'GPS-S1C']
e1 = satellite_data_all_frequencies.loc[satellite_data_all_frequencies['OB_Type'] == 'GAL-E1']
glo_s1c = satellite_data_all_frequencies.loc[satellite_data_all_frequencies['OB_Type'] == 'GLO-S1C']

#L2 Band:
s2x = satellite_data_all_frequencies.loc[satellite_data_all_frequencies['OB_Type'] == 'GPS-S2X']
glo_s2c = satellite_data_all_frequencies.loc[satellite_data_all_frequencies['OB_Type'] == 'GLO-S2C']

#L5 Band:
s5x = satellite_data_all_frequencies.loc[satellite_data_all_frequencies['OB_Type'] == 'GPS-S5X']
e5 = satellite_data_all_frequencies.loc[satellite_data_all_frequencies['OB_Type'] == 'GAL-E5a']

#invert reflector heights + distance above water level:
all_freq['estimated_reflector_height'] = all_freq['estimated_reflector_height'].apply(lambda x: x * -1 + 3.00)

#L1 Band:
s1c['estimated_reflector_height'] = s1c['estimated_reflector_height'].apply(lambda x: x * -1.013 + 3.00)
e1['estimated_reflector_height'] = e1['estimated_reflector_height'].apply(lambda x: x * -1.013 + 3.00)
glo_s1c['estimated_reflector_height'] = glo_s1c['estimated_reflector_height'].apply(lambda x: x * -1.013 + 3.00)

#L2 Band:
s2x['estimated_reflector_height'] = s2x['estimated_reflector_height'].apply(lambda x: x * -1.015 + 3.00)
glo_s2c['estimated_reflector_height'] = glo_s2c['estimated_reflector_height'].apply(lambda x: x * -1.015 + 3.00)

#L5 Band:
s5x['estimated_reflector_height'] = s5x['estimated_reflector_height'].apply(lambda x: x * -1.017 + 3.00)
e5['estimated_reflector_height'] = e5['estimated_reflector_height'].apply(lambda x: x * -1.017 + 3.00)

#------------------------------------------------RMSE Analysis----------------------------------------------------------
# calculate RMSE:
#calculate bias
bias_all = calculate_bias(all_freq['tg_sea_level'], all_freq['estimated_reflector_height'])

#L1 Band:
bias_S1C = calculate_bias(s1c['tg_sea_level'], s1c['estimated_reflector_height'])
bias_E1 = calculate_bias(e1['tg_sea_level'], e1['estimated_reflector_height'])
bias_glo_S1C = calculate_bias(glo_s1c['tg_sea_level'], glo_s1c['estimated_reflector_height'])

#L2 Band:
bias_S2X = calculate_bias(s2x['tg_sea_level'], s2x['estimated_reflector_height'])
bias_S2C = calculate_bias(glo_s2c['tg_sea_level'], glo_s2c['estimated_reflector_height'])

#L5 Band:
bias_S5X = calculate_bias(s5x['tg_sea_level'], s5x['estimated_reflector_height'])
bias_E5 = calculate_bias(e5['tg_sea_level'], e5['estimated_reflector_height'])

#apply rmse_function to dataframe:
#All frequencies->
rmse_all = rmse_function(all_freq['tg_sea_level'] - bias_all, all_freq['estimated_reflector_height'])

#L1 Band:
rmse_s1c = rmse_function(s1c['tg_sea_level'] - bias_S1C, s1c['estimated_reflector_height'])
rmse_e1 = rmse_function(e1['tg_sea_level'] - bias_E1, e1['estimated_reflector_height'])
rmse_glo_s1c = rmse_function(glo_s1c['tg_sea_level'] - bias_glo_S1C, glo_s1c['estimated_reflector_height'])

#L2 Band:
rmse_s2x = rmse_function(s2x['tg_sea_level'] - bias_S2X, s2x['estimated_reflector_height'])
rmse_s2c = rmse_function(glo_s2c['tg_sea_level'] - bias_S2C, glo_s2c['estimated_reflector_height'])

#L5 Band:
rmse_s5x = rmse_function(s5x['tg_sea_level'] - bias_S5X, s5x['estimated_reflector_height'])
rmse_e5 = rmse_function(e5['tg_sea_level'] - bias_E5, e5['estimated_reflector_height'])

#print(rmse_all, rmse_s1c, rmse_e1, rmse_glo_s1c, rmse_s2x, rmse_s2c, rmse_s5x, rmse_e5)
#------------------------------------------------Correlation------------------------------------------------------------
# correlation values
corr_all = all_freq['tg_sea_level'].corr(all_freq['estimated_reflector_height'])

#L1 Band:
corr_S1C = s1c['tg_sea_level'].corr(s1c['estimated_reflector_height'])
corr_E1 = e1['tg_sea_level'].corr(e1['estimated_reflector_height'])
corr_glo = glo_s1c['tg_sea_level'].corr(glo_s1c['estimated_reflector_height'])

#L2 Band:
corr_S2X = s2x['tg_sea_level'].corr(s2x['estimated_reflector_height'])
corr_S2C = glo_s2c['tg_sea_level'].corr(glo_s2c['estimated_reflector_height'])

#L5 Band:
corr_S5X = s5x['tg_sea_level'].corr(s5x['estimated_reflector_height'])
corr_E5 = e5['tg_sea_level'].corr(e5['estimated_reflector_height'])

#print(corr_all, corr_S1C, corr_E1, corr_glo, corr_S2X, corr_S2C, corr_S5X, corr_E5)
#------------------------------------------------Create Plots-----------------------------------------------------------

#choose color scheme:
cmap = plt.cm.get_cmap('tab20c')

#create subplots:
fig = plt.figure(figsize = (14,5))
ax1 = fig.add_subplot(111)

#plot tide gauge:
ax1.plot(all_freq['time'], all_freq['tg_sea_level'], color='k')  #for combined frequencies figure
#ax1.plot(all_freq['time'], all_freq['tg_sea_level'], color='k') #for showing individual frequncies

#plot satellites:
#L5 Band:
#ax1.plot(s5x['time'], s5x['estimated_reflector_height'], color=cmap(8))
#ax1.plot(e5['time'], e5['estimated_reflector_height'], color=cmap(9))

#L2 Band:
#ax1.plot(s2x['time'], s2x['estimated_reflector_height'], color=cmap(4))
#ax1.plot(glo_s2c['time'], glo_s2c['estimated_reflector_height'], color=cmap(5))

#L1 Band:
#ax1.plot(s1c['time'], s1c['estimated_reflector_height'], color=cmap(0))
#ax1.plot(e1['time'], e1['estimated_reflector_height'], color=cmap(1))
#ax1.plot(glo_s1c['time'], glo_s1c['estimated_reflector_height'], color=cmap(2))

#All Bands:
#ax1.plot(all_freq['time'], all_freq['estimated_reflector_height'], color=cmap(4)) #for combined frequencies figure
ax1.plot(all_freq['time'], all_freq['estimated_reflector_height'], color = 'r') #for showing individual frequncies

#ax2.set_xlabel("Date")
ax1.set_ylabel("Sea Level [m]", size=12)
ax1.set_xlabel("Date [yyyy-mm-dd]", size=12)
ax1.set_title('6 Hour Window Tide Gauge Measurements vs Combined Frequencies (GPS + Galielo + GLONASS)', size=14)
#ax1.set_title('6 Hour Window Tide Gauge Measurements vs All Frequencies (High Winds)', size=14)
#plot legend
ax1.legend(['Tide Gauge', 'Multiple Constellations'], loc='upper left')
#ax1.legend(['Tide Gauge', 'GPS-S5X', 'E5a', 'GPS-S2X', 'GLO-S2C', 'GPS-S1C', 'E1', 'GLO-S1C', 'All Frequencies'],
           #ncol=2, loc='upper left')
ax1.grid()

#ax2.set_xlabel("Date")
ax1.set_ylabel("Sea Level [m]")
fig.tight_layout()
# Create second axes, the top-left plot with orange plot

fig2, ax23 = plt.subplots(1, 2, figsize=(14, 5))
sub2, sub3 = ax23.flatten()
#fig2 = plt.figure()
#sub2 = fig2.add_subplot(1,1,1) # second row, second column, third cell
x_axis = np.arange(0, 1)
sub2.bar(x_axis, rmse_all, color='r', width=1/9, label='Combined')

#L1 Band:
sub2.bar(x_axis + 1/8, rmse_s1c, color=cmap(0), width=1/9, label='GPS-S1C')
sub2.bar(x_axis + 2/8, rmse_e1, color=cmap(1), width=1/9, label='E1')
sub2.bar(x_axis + 3/8, rmse_glo_s1c, color=cmap(2), width=1/9, label='GLO-S1C')

#L2 Band:
sub2.bar(x_axis + 4/8, rmse_s2x, color=cmap(4), width=1/9, label='GPS-S2X')
sub2.bar(x_axis + 5/8, rmse_s2c, color=cmap(5), width=1/9, label='GLO-S2C')

#L5 Band:
sub2.bar(x_axis + 6/8, rmse_s5x, color=cmap(8), width=1/9, label='S5X')
sub2.bar(x_axis + 7/8, rmse_e5, color=cmap(9), width=1/9, label='E5a')

#label x and y axis:
sub2.set_xlabel("Frequency", size=12)
sub2.set_ylabel("RMSE [m]", size=12)
sub2.set_title('RMSE Results', size=12)

#plot legend
#ax3.legend(['S125', 'S1C', 'S2X', 'S5X'])
x_ticks = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875 ]
x_labels = ['ALL', 'GPS-S1C', 'E1', 'GLO-S1C', 'GPS-S2X', 'GLO-S2C', 'GPS-S5X', 'E5a']

#add x-axis values to plot
#plt.xticks(ticks=x_ticks, labels=x_labels)
sub2.set_xticks(ticks=x_ticks, labels=x_labels, rotation=45)

# Create third axes, a combination of third and fourth cell
#fig3 = plt.figure()
#sub3 = fig3.add_subplot(1,1,1) # second row, second colum, fourth cell
x_axis = np.arange(0, 1)

#All Frequencies:
sub3.bar(x_axis, np.mean(all_freq['no_of_good_obs']/1.5), color='r', width=1/9, label='Combined')

#L1 Band:
sub3.bar(x_axis + 1/8, np.mean(s1c['no_of_good_obs']), color=cmap(0), width=1/9, label='GPS-S1C')
sub3.bar(x_axis + 2/8, np.mean(e1['no_of_good_obs']), color=cmap(1), width=1/9, label='E1')
sub3.bar(x_axis + 3/8, np.mean(glo_s1c['no_of_good_obs']), color=cmap(2), width=1/9, label='GLO-S1C')

#L2 Band:
sub3.bar(x_axis + 4/8, np.mean(s2x['no_of_good_obs']), color=cmap(4), width=1/9, label='GPS-S2X')
sub3.bar(x_axis + 5/8, np.mean(glo_s2c['no_of_good_obs']), color=cmap(5), width=1/9, label='GLO-S2C')

#L5 Band:
sub3.bar(x_axis + 6/8, np.mean(s5x['no_of_good_obs']), color=cmap(8), width=1/9, label='S5X')
sub3.bar(x_axis + 7/8, np.mean(e5['no_of_good_obs']), color=cmap(9), width=1/9, label='E5a')

#label x and y axis:
sub3.set_xlabel("Frequency", size=12)
sub3.set_ylabel("Number of Observations", size=12)
sub3.set_title('Average Number of Observations per Epoch', size=14)

#plot legend:
x_ticks = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
x_labels = ['ALL', 'GPS-S1C', 'E1', 'GLO-S1C', 'GPS-S2X', 'GLO-S2C', 'GPS-S5X', 'E5a']

#add x-axis values to plot
plt.xticks(ticks=x_ticks, labels=x_labels)
plt.xticks(rotation=45)

fig2.tight_layout()

# making subplots
fig3, ax = plt.subplots(1, 2, figsize=(14, 5))
ax2, ax3 = ax.flatten()

fig4, axb = plt.subplots(1, 2, figsize=(14, 5))
ax4, ax5 = axb.flatten()

fig5, axc = plt.subplots(1, 2, figsize=(14, 5))
ax6, ax7 = axc.flatten()

fig6, axd = plt.subplots(1, 2, figsize=(14, 5))
ax8, ax9 = axd.flatten()

#create regression lines:
#All freq
all_regression_line = regression_line(all_freq['estimated_reflector_height'], all_freq['tg_sea_level'])

#L1 Band:
s1c_regression_line = regression_line(s1c['estimated_reflector_height'], s1c['tg_sea_level'])
e1_regression_line = regression_line(e1['estimated_reflector_height'], e1['tg_sea_level'])
glo_s1c_regression_line = regression_line(glo_s1c['estimated_reflector_height'], glo_s1c['tg_sea_level'])

#L2 Band:
s2x_regression_line = regression_line(s2x['estimated_reflector_height'], s2x['tg_sea_level'])
glo_s2c_regression_line = regression_line(glo_s2c['estimated_reflector_height'], glo_s2c['tg_sea_level'])

#L3 Band:
s5x_regression_line = regression_line(s5x['estimated_reflector_height'], s5x['tg_sea_level'])
e5_regression_line = regression_line(e5['estimated_reflector_height'], e5['tg_sea_level'])

# set data with subplots and plot
#All freq
ax2.scatter(all_freq['estimated_reflector_height']+0.2, all_freq['tg_sea_level'],
                 c=all_freq['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'), s=0.9)
ax2.plot(all_freq['estimated_reflector_height']+0.2, all_regression_line, color='k')

#L1 Band
ax3.scatter(s1c['estimated_reflector_height']+0.2, s1c['tg_sea_level'],
                 c=s1c['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'), s=0.9)
ax4.scatter(e1['estimated_reflector_height']+0.2, e1['tg_sea_level'],
                 c=e1['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'), s=0.9)
ax5.scatter(glo_s1c['estimated_reflector_height']+0.2, glo_s1c['tg_sea_level'],
                 c=glo_s1c['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'), s=0.9)

ax3.plot(s1c['estimated_reflector_height']+0.2, s1c_regression_line, color='k')
ax4.plot(e1['estimated_reflector_height']+0.2, e1_regression_line, color='k')
ax5.plot(glo_s1c['estimated_reflector_height']+0.2, glo_s1c_regression_line, color='k')

#L2 Band
ax6.scatter(s2x['estimated_reflector_height']+0.2, s2x['tg_sea_level'],
                 c=s2x['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'), s=0.9)
ax7.scatter(glo_s2c['estimated_reflector_height']+0.2, glo_s2c['tg_sea_level'],
                 c=glo_s2c['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'), s=0.9)

ax6.plot(s2x['estimated_reflector_height']+0.2, s2x_regression_line, color='k')
ax7.plot(glo_s2c['estimated_reflector_height']+0.2, glo_s2c_regression_line, color='k')

#L5 band
ax8.scatter(s5x['estimated_reflector_height']+0.2, s5x['tg_sea_level'],
                 c=s5x['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'), s=0.9)
ax9.scatter(e5['estimated_reflector_height']+0.2, e5['tg_sea_level'],
                 c=e5['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'), s=0.9)

ax8.plot(s5x['estimated_reflector_height']+0.2, s5x_regression_line, color='k')
ax9.plot(e5['estimated_reflector_height']+0.2, e5_regression_line, color='k')

ax2.set_title("All Frequencies")
ax3.set_title("GPS-S1C")
ax4.set_title("GAL-E1")
ax5.set_title("GLO-S1C")
ax6.set_title("GPS-S2X")
ax7.set_title("GLO-S2C")
ax8.set_title("GPS-S5X")
ax9.set_title("GAL-E5a")

ax2.set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")
ax3.set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")
ax4.set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")
ax5.set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")
ax6.set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")
ax7.set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")
ax8.set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")
ax9.set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")

ax2.grid()
ax3.grid()
ax4.grid()
ax5.grid()
ax6.grid()
ax7.grid()
ax8.grid()
ax9.grid()

#fig3.tight_layout()
norm = mpl.colors.Normalize(vmin=0, vmax=2)
fig3.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.get_cmap('RdYlBu_r')), ax=ax.ravel(), orientation='vertical',
                                   label='Estimated Reflector Height STD')

# set spacing
norm = mpl.colors.Normalize(vmin=0, vmax=2)
fig4.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.get_cmap('RdYlBu_r')), ax=axb.ravel(), orientation='vertical',
                                   label='Estimated Reflector Height STD')

# set spacing
norm = mpl.colors.Normalize(vmin=0, vmax=2)
fig5.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.get_cmap('RdYlBu_r')), ax=axc.ravel(), orientation='vertical',
                                   label='Estimated Reflector Height STD')

# set spacing
norm = mpl.colors.Normalize(vmin=0, vmax=2)
fig6.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.get_cmap('RdYlBu_r')), ax=axd.ravel(), orientation='vertical',
                                   label='Estimated Reflector Height STD')


fig.savefig('/home/OceanJasper/GNSS-R/altimetry/final_thesis_figures/gnss/6_hour/6_hour_timeseries_comb_freq_ieee')
fig2.savefig('/home/OceanJasper/GNSS-R/altimetry/final_thesis_figures/gnss/6_hour/6_hour_rmse_epoch')
#fig3.savefig('/home/OceanJasper/GNSS-R/altimetry/final_thesis_figures/gnss/6_hour/6_hour_corr_all_l1_rl')
#fig4.savefig('/home/OceanJasper/GNSS-R/altimetry/final_thesis_figures/gnss/6_hour/6_hour_corr_l1_rl')
#fig5.savefig('/home/OceanJasper/GNSS-R/altimetry/final_thesis_figures/gnss/6_hour/6_hour_corr_l2_rl')
#fig6.savefig('/home/OceanJasper/GNSS-R/altimetry/final_thesis_figures/gnss/6_hour/6_hour_corr_l3_rl')

plt.show()








