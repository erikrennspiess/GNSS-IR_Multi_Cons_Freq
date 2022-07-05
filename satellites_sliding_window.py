import pandas as pd
import numpy as np
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
sat_meas = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/orbit_output/2018-09-01_2018-09-30.csv')

#convert time column to datetime
sat_meas['time'] = pd.to_datetime(sat_meas['time'])

#Join datasets:
meas = pd.merge(sat_meas, tide_gauge_date_selection, how="left", on=["time"])

target_obs_types = ['S1C', 'S2X', 'S5X']
#target_obs_types = ['S5X']


meas_selected = meas.loc[meas['obs_type'].isin(target_obs_types)]
time_array = pd.to_datetime(meas['time'])
start_time = np.min(time_array).round(freq='H')
end_time = np.max(time_array).round(freq='H')

step_size = 300 #sec
step_size = pd.to_timedelta(step_size,unit='s')
steps_array = pd.date_range(start=start_time, end=end_time, freq=step_size)

window_size = 3 * 3600 #sec
window_size = pd.to_timedelta(window_size,unit='s')

prn_results = []
for t in steps_array:
    ind = abs(time_array - t) <= window_size / 2
    meas_in_window = meas_selected.loc[ind]
    for sv, i in meas_in_window.groupby('sv'):
        obs_std = np.std(i['reflector_height'])
        outliers_ind = np.abs(i['reflector_height'] - np.median(i['reflector_height'])) >= 2 * obs_std
        meas_good = i[~outliers_ind]
        estimated_reflector_height = np.median(meas_good['reflector_height'])
        tg_sea_level = np.mean(i['sea_level'])
        prn_results.append([t, sv, obs_std, estimated_reflector_height, tg_sea_level])

    satellite_measurements = pd.DataFrame(prn_results,
                                          columns=['Time', 'PRN', 'Obs_Std', 'Reflector_Height', 'TG_Height'])
print(satellite_measurements)

file_name = '3hour_window_sat_all.csv'
#write file & name file:
if os.path.exists('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/satellites/' + file_name):
    print("File is already in directory")
else:
    satellite_measurements.to_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/satellites/'
                                  + file_name, mode='a', index=False, header=True)
    print("Created file")







