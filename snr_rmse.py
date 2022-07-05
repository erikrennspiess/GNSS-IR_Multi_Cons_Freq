#------------------------------------------------Program_Details--------------------------------------------------------
# Created by: Rennspiess, E., Hoseini, M.
# Created on: Nov, 18, 2021
# Updated on: Dec, 17, 2021
# Error analysis for comparing GNSS-R SNR data with Tide Gauge data from Onsala, Sweden. Produces a bar graph figure
# & time series figure.
#------------------------------------------------Libraries--------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
#------------------------------------------------Functions--------------------------------------------------------------
#inverse modelling of SNR data:
def inverse_snr(A, h,l_ambda, elevation_angle):
    height_wavelength = (4*np.pi*h)//l_ambda
    sin_e = height_wavelength*np.sin(elevation_angle)
    return A*np.sin(sin_e)

#Root Mean Squared Error (RMSE) window function:
def rmse_function(snr, sea_level_tg):
    return 1/np.sqrt(((snr - sea_level_tg) ** 2).mean())

#correlation function:
def correlation_function(reflector_height, sea_level_tg):
    return np.corrcoef(reflector_height, sea_level_tg)
#------------------------------------------------Data-------------------------------------------------------------------
#column names:
#time, sv, obs_type, mean_elevation, mean_azimuth, oscillation_power, reflector_height, Datum Tid (UTC). sea_level_tg
#Kvalitet, Mätdjup (m), Unnamed: 4, Tidsutsnitt
#-----------------------------------------------------------------------------------------------------------------------
#import data set:
sat_data = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/orbit_output/2018-09-01_2018-09-30.csv')
tide_gauge_data = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/tide_gauge_data/tg_data.txt', sep=';')

#Pre-processing for sat and tide gauge data:
sat_data['time'] = pd.to_datetime(sat_data['time'])
tide_gauge_data['sea_level'] = pd.to_datetime(tide_gauge_data['sea_level'])

#select date range:
start_date = '2018-09-01 00:00:00'
end_date = '2018-09-30 23:59:00'

#apply date masks to data frame:
date_mask = (tide_gauge_data['sea_level'] > start_date) & (tide_gauge_data['sea_level'] <= end_date)
tide_gauge_data = tide_gauge_data.loc[date_mask]

#combine gnssr & tg data:
sat_tg = pd.merge(sat_data, tide_gauge_data, how='inner', left_on='time', right_on='sea_level')
sat_tg= sat_tg.rename(columns={'sea_level': 'sea_level_tg'})

#add wavelength for each observation type:
#set conditions:
conditions = [(sat_tg['obs_type'] == 'S1C'), (sat_tg['obs_type'] == 'S2X'), (sat_tg['obs_type'] == 'S5X')]
#create values:
signal_wavelength = [0.1905, 0.2445, 0.2548]  # wavelengths of GPS L1,L2 & L5 in meters, must correspond to obs_types
#create new column w/ values:
sat_tg['wavelength'] = np.select(conditions, signal_wavelength)

#plot grouped bar figure:
rmse_results_pivot = pd.pivot_table(
    rmse_results,
    values="RMSE_[m]",
    index="satellite",
    columns="obs_type"
)
# Plot a bar chart using the DF
ax = rmse_results_pivot.plot(kind="bar")
# Get a Matplotlib figure from the axes object for formatting purposes
fig = ax.get_figure()
# Change the plot dimensions (width, height)
fig.set_size_inches(7, 6)
# Change the axes labels
ax.set_xlabel("Satellite")
ax.set_ylabel("RMSE [cm]")

plt.title('RMSE: GNSSR Reflector Height vs TG Sea Level')
plt.legend()
plt.show()

#------------------------------------------------Inverse_modelling_of_SNR_Data------------------------------------------
snr_tg_wl = []
for observation, i in sat_tg.groupby('obs_type'):
    time = i.time
    sat = i.sv
    tg = i.sea_level_tg/10
    amplitude = i.oscillation_power
    height = i.reflector_height
    l_ambda = i.wavelength
    elevation_angle = i.mean_elevation
    water_level_snr = inverse_snr(amplitude, height, l_ambda, elevation_angle)
    snr_height_df = pd.DataFrame({"dtime": time, "sat": sat, "snr_wl": water_level_snr, "tg_wl": tg})
    snr_height_df["obs_type"] = observation
    snr_tg_wl.append(snr_height_df)

water_levels = pd.concat(snr_tg_wl)
#------------------------------------------------SNR Least Squares Adjustment-------------------------------------------
wl_smoothed_values_df = []
for i in water_levels:
    time = water_levels['dtime']
    sat = water_levels['sat']
    obs_type = water_levels['obs_type']
    tide_gauge = water_levels['tg_wl']
    y_hat = savgol_filter(water_levels['snr_wl'], 51, 3) # window size 51 (4 hours, 15 minutes) , polynomial order 3
    adjusted_df = pd.DataFrame({"dtime": time,"satellite": sat, "obs_type": obs_type,
                                "y_hat": y_hat, "tg_wl": tide_gauge})
    wl_smoothed_values_df.append(adjusted_df)

adjusted_snr_values = pd.concat(wl_smoothed_values_df)
#drop duplicate data points:
adjusted_snr_values = adjusted_snr_values.drop_duplicates()

#unit converstions and calculations:
for observation, i in adjusted_snr_values.groupby('obs_type'):
    adjusted_snr_values['tg_wl'] = (adjusted_snr_values['tg_wl'] - adjusted_snr_values['tg_wl'].mean())
    adjusted_snr_values['y_hat'] = (adjusted_snr_values['y_hat']-adjusted_snr_values['y_hat'].mean())*-1 #inverse porportion

adjusted_snr_values = adjusted_snr_values.sort_values(['dtime'])
#------------------------------------------------Error_Analysis---------------------------------------------------------
#column names: time, satellite, snr_wl, y_hat, moving_avg, tg_wl, obs_type
rmse_df = []
for observation, i in adjusted_snr_values.groupby('obs_type'):
    for satellite, j in i.groupby('satellite'):
        rmse_value = rmse_function(j['y_hat'], (j['tg_wl']))
        rmse_correlation = correlation_function(j['y_hat'], (j['tg_wl']))
        rmse_df.append([satellite, observation, rmse_value, rmse_correlation[0, 1]])

rmse_results = pd.DataFrame(rmse_df, columns=['satellite', 'obs_type', 'RMSE_[m]', 'RMSE_corr'])

#-----------------------------------------------RMSE Plotting-----------------------------------------------------------

#-----------------------------------------------Moving Median-----------------------------------------------------------
S1C = adjusted_snr_values.loc[(adjusted_snr_values['obs_type'] == 'S1C')]
S1C['timestamp'] = (S1C['dtime'] - pd.Timestamp("1970-01-01")) //pd.Timedelta('1s')

# moving step and time window for selecting observations
moving_window = 3600 # seconds
moving_step = 60  # seconds

start_obs_time = np.min(S1C['dtime'])
end_obs_time = np.max(S1C['dtime'])

median_values_df=[]
for i in pd.date_range(start=start_obs_time, end=end_obs_time, freq=str(moving_step) + "S"):
    ind_time = (S1C['dtime'] >= i - pd.Timedelta(str(moving_window / 2) + "s")) & \
               (S1C['dtime'] <= i + pd.Timedelta(str(moving_window / 2) + "s"))
    #select indexes based on condition:
    selected_obs = S1C[ind_time]
    #set variables:
    selected_timestamp = selected_obs['timestamp']
    selected_tide_gauge_obs = selected_obs['tg_wl']
    selected_snr = selected_obs['y_hat']
    #perform outlier filter:
    std_selected_snr = np.std(selected_snr)
    filter_good_observations = np.abs(selected_snr - np.mean(selected_snr)) <= 2.5 * std_selected_snr
    final_sea_level = np.median(selected_snr[filter_good_observations])
    std_final_sea_level = np.std(selected_snr[filter_good_observations])
    average_reflector_height = np.mean(selected_snr[filter_good_observations])
    median_tide_gauge_water_level = np.median(selected_tide_gauge_obs)
    filtered_timestamp = np.mean(selected_timestamp[filter_good_observations])
    #convert sea levels to reference frame(H=-N+h):

    # save to data frame:
    median_values_df.append([filtered_timestamp, final_sea_level,std_final_sea_level,
                             average_reflector_height, median_tide_gauge_water_level])

final_sea_levels = pd.DataFrame(median_values_df, columns=['date_time', 'sea_level','sea_level_std',
                                                           'avg_reflector_height','tide_gauge'])
final_sea_levels['date_time'] = pd.to_datetime(final_sea_levels['date_time'], unit='s')
#-----------------------------------------------Moving Average----------------------------------------------------------



#------------------------------------------------save data--------------------------------------------------------------
#write file & name file:
#out_variable = '/home/erikrennspiess/Desktop/gnssr_programs/altimetry/'
#file_name = 'final_sea_levels_sep_2018'
#if os.path.exists(out_variable+file_name):
    #print("File is already in directory")
#else:
    #final_sea_levels.to_csv(out_variable+'/'+file_name, mode='a', index=False, header=True)
    #print("Created file")


















