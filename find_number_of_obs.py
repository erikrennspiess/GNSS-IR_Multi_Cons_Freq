import statistics
import pandas as pd
import numpy as np
import glob
import os
#------------------------------------------------Data-------------------------------------------------------------------
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

#SATELLITE:
sat_meas = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/orbit_output/2018-09-01_2018-09-30_glonass.csv')
sat_meas = sat_meas.loc[sat_meas['mean_azimuth'] > 75]


#convert time column to datetime
sat_meas['time'] = pd.to_datetime(sat_meas['time'])

#Join datasets:
meas = pd.merge(sat_meas, tide_gauge_date_selection, how="left", on=["time"])

#target_obs_types = ['S1C', 'S2X', 'S5X']    #gps
#target_obs_types = ['S1X', 'S5X']           #galileo
#target_obs_types = ['S1C', 'S2C']            #glonass
target_obs_types = ['S2C']


meas_selected = meas.loc[meas['obs_type'].isin(target_obs_types)]
time_array = pd.to_datetime(meas['time'])
start_time = np.min(time_array).round(freq='H')
end_time = np.max(time_array).round(freq='H')

step_size = 300 #sec
step_size = pd.to_timedelta(step_size,unit='s')
steps_array = pd.date_range(start=start_time, end=end_time, freq=step_size)

window_size = 3 * 3600 #sec
window_size = pd.to_timedelta(window_size,unit='s')

results = []
for t in steps_array:
    ind = abs(time_array-t)<=window_size/2
    meas_in_window = meas_selected.loc[ind]
    obs_std = np.std(meas_in_window['reflector_height'])
    outliers_ind = np.abs( meas_in_window['reflector_height'] - np.median(meas_in_window['reflector_height'])) >= 2 * obs_std
    meas_good = meas_in_window[~outliers_ind]
    estimated_reflector_height = np.median(meas_good['reflector_height'])
    estimated_reflector_height_std = np.std(meas_good['reflector_height'])
    no_of_good_obs = meas_good['reflector_height'].size
    no_of_outliers = np.sum(outliers_ind)
    no_of_sats = np.unique(meas_good['sv']).size
    prn = np.unique(meas_good['sv'])
    tg_sea_level = np.mean(meas_in_window['sea_level'])
    results.append([t, np.round(estimated_reflector_height, decimals=3),
                    np.round(estimated_reflector_height_std, decimals=3), tg_sea_level, no_of_good_obs,
                    no_of_outliers, no_of_sats, prn])


results = pd.DataFrame(results, columns=['time', 'estimated_reflector_height', 'estimated_reflector_height_std',
                                         'tg_sea_level', 'no_of_good_obs', 'no_of_outliers', 'no_of_sats', 'prn'])
print(results)
file_name = '3_hour_window_gps_glonass_S2C.csv'
#write file & name file:
if os.path.exists('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/frequencies/Glonass_frequencies/' + file_name):
    print("File is already in directory")
else:
    results.to_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/frequencies/Glonass_frequencies/'
                   + file_name, mode='a', index=False, header=True)
    print("Created file")



