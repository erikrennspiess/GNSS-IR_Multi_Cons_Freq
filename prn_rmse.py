#------------------------------------------------Program_Details--------------------------------------------------------
# Created by: Rennspiess, E., Hoseini, M.
# Created on: March, 27, 2022
# Updated on: March, 27, 2022
# Error analysis for comparing GNSS-R SNR data with Tide Gauge data from Onsala, Sweden.
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

#------------------------------------------------Functions--------------------------------------------------------------
#Root Mean Squared Error (RMSE) window function:
def rmse_function(tide_gauge, snr):
    MSE = np.square(np.subtract(tide_gauge, snr)).mean()
    RMSE = math.sqrt(MSE)
    return RMSE

#correlation function:
def correlation_function(sea_level_tg, reflector_height):
    return np.corrcoef(reflector_height, sea_level_tg)

#------------------------------------------------Data-------------------------------------------------------------------
#SATELLITES & TIDEGAUGE DATA:
reflector_all = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/satellites/6hour_window_sat_all.csv')
reflector_s1c = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/satellites/6hour_window_sat_S1C.csv')
reflector_s2x = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/satellites/6hour_window_sat_S2X.csv')
reflector_s5x = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/satellites/6hour_window_sat_S5X.csv')

for col in reflector_all.columns:
    print(col)

#convert time column to datetime
reflector_all['Time'] = pd.to_datetime(reflector_all['Time'])
reflector_s1c['Time'] = pd.to_datetime(reflector_s1c['Time'])
reflector_s2x['Time'] = pd.to_datetime(reflector_s2x['Time'])
reflector_s5x['Time'] = pd.to_datetime(reflector_s5x['Time'])

#invert reflector heights + distance above water level:
reflector_all['Reflector_Height'] = reflector_all['Reflector_Height'] * -1 + 3.00
reflector_s1c['Reflector_Height'] = reflector_s1c['Reflector_Height'] * -1 + 3.00
reflector_s2x['Reflector_Height'] = reflector_s2x['Reflector_Height'] * -1 + 3.00
reflector_s5x['Reflector_Height'] = reflector_s5x['Reflector_Height'] * -1 + 3.00



#------------------------------------------------Error_Analysis---------------------------------------------------------
#calculate bias:
diff_all = reflector_all['TG_Height'] - reflector_all['Reflector_Height']
bias_all = diff_all.mean()

#S1C:
diff_S1C = reflector_s1c['TG_Height'] - reflector_s1c['Reflector_Height']
bias_S1C = diff_S1C.mean() * 1.013

#S2X:
diff_S2X = reflector_s2x['TG_Height'] - reflector_s2x['Reflector_Height']
bias_S2X = diff_S2X.mean() * 1.015

#S5X:
diff_S5X = reflector_s5x['TG_Height'] - reflector_s5x['Reflector_Height']
bias_S5X = diff_S5X.mean() * 1.017

#apply rmse function and remove bias:
#add labels to rows:
reflector_s1c['Ob_Type'] = 'S1C'
reflector_s2x['Ob_Type'] = 'S2X'
reflector_s5x['Ob_Type'] = 'S5X'
#S125:
all_df = []
for satellite, i in reflector_all.groupby('PRN'):
    tidegauge = i['TG_Height'] - bias_all
    single_rmse_value = rmse_function(tidegauge, i.Reflector_Height)
    all_df.append([satellite,single_rmse_value ])
all_df_results = pd.DataFrame(all_df, columns=['PRN', 'RMSE'])
#add label:
all_df_results['Ob_Type'] = 'S125'

#S1C:
S1C_df = []
for satellite, i in reflector_s1c.groupby('PRN'):
    tidegauge = i['TG_Height'] - bias_S1C
    single_rmse_value = rmse_function(tidegauge, i.Reflector_Height)
    S1C_df.append([satellite,single_rmse_value ])
S1C_df_results = pd.DataFrame(S1C_df, columns=['PRN', 'RMSE'])
#add label:
S1C_df_results['Ob_Type'] = 'S1C'

#S2X:
S2X_df = []
for satellite, i in reflector_s2x.groupby('PRN'):
    tidegauge = i['TG_Height'] - bias_S2X
    single_rmse_value = rmse_function(tidegauge, i.Reflector_Height)
    S2X_df.append([satellite,single_rmse_value ])
S2X_df_results = pd.DataFrame(S2X_df, columns=['PRN', 'RMSE'])
#add label:
S2X_df_results['Ob_Type'] = 'S2X'

#S5X:
S5X_df = []
for satellite, i in reflector_s5x.groupby('PRN'):
    tidegauge = i['TG_Height'] - bias_S5X
    single_rmse_value = rmse_function(tidegauge, i.Reflector_Height)
    S5X_df.append([satellite,single_rmse_value ])
S5X_df_results = pd.DataFrame(S5X_df, columns=['PRN', 'RMSE'])
#add label:
S5X_df_results['Ob_Type'] = 'S5X'

cmap = plt.cm.get_cmap('tab20c')

fig = plt.figure(111)
ax = fig.add_subplot(111)
x_axis = np.arange(len(all_df_results['PRN']))
ax.bar(x_axis, all_df_results['RMSE'], color=cmap(0), width=1/5, linewidth=0, label='S125')
ax.bar(x_axis + 1/4, S1C_df_results['RMSE'], color=cmap(4), width=1/5, linewidth=0, label='S1C')
ax.bar(x_axis + 2/4, S2X_df_results['RMSE'], color=cmap(5), width=1/5, linewidth=0, label='S2X')
ax.bar(x_axis + 3/4, S5X_df_results['RMSE'], color=cmap(6), width=1/5, linewidth=0, label='S5X')

#label/set title, x and y axis:
ax.set_xlabel('PRN')
ax.set_ylabel('Sea Level [m]')
ax.set_title('6 Hour Window RMSE: PRN & Frequency')
ax.set_xticks(x_axis + 1/2)
ax.set_xticklabels(all_df_results.PRN.unique())
#plot legend
ax.legend(['S125', 'S1C', 'S2X', 'S5X'])
plt.show()