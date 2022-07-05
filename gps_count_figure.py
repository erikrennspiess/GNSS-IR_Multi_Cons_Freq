import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches

#------------------------------------------------Define Functions-------------------------------------------------------
def clean_alt_list(list_):
    list_ = list_.replace(' ', ',')
    return list_

def to_1D(series):
 return pd.Series([x for _list in series for x in _list])


def boolean_df(item_lists, unique_items):
    # Create empty dict
    bool_dict = {}

    # Loop through all the tags
    for i, item in enumerate(unique_items):
        # Apply boolean mask
        bool_dict[item] = item_lists.apply(lambda x: item in x)

    # Return the results as a dataframe
    return pd.DataFrame(bool_dict)

def sort_satellites(prn_index, prn_values):
    prn_dataframe = pd.DataFrame({'prn': prn_index, 'value': prn_values})
    prn_dataframe['sort_2'] = prn_dataframe['prn'].str.extract('(\d+)', expand=False).astype(int)
    prn_dataframe.sort_values('sort_2', inplace=True, ascending=True)
    prn_dataframe = prn_dataframe.drop('sort_2', axis=1)

    return prn_dataframe

#------------------------------------------------Import Data------------------------------------------------------------
reflector_s1c = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/frequencies/GPS_frequencies/'
                            '6hour_window_S1C_azi.csv')

reflector_s2x = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/frequencies/GPS_frequencies/'
                              '6hour_window_s2x_azi.csv')

reflector_s5x = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/frequencies/GPS_frequencies/'
                              '6hour_window_s5x_azi.csv')

#------------------------------------------------Process Data-----------------------------------------------------------
#S1C
reflector_s1c["prn"] = reflector_s1c["prn"].apply(clean_alt_list)
reflector_s1c["prn"] = reflector_s1c["prn"].apply(eval)

prn_index_s1c = to_1D(reflector_s1c["prn"]).value_counts().index
prn_values_s1c = to_1D(reflector_s1c["prn"]).value_counts().values

prn_dataframe_s1c = sort_satellites(prn_index_s1c, prn_values_s1c)
prn_dataframe_s1c['obs_type'] = 'S1C'

#S2X
reflector_s2x["prn"] = reflector_s2x["prn"].apply(clean_alt_list)
reflector_s2x["prn"] = reflector_s2x["prn"].apply(eval)

prn_index_s2x = to_1D(reflector_s2x["prn"]).value_counts().index
prn_values_s2x = to_1D(reflector_s2x["prn"]).value_counts().values

prn_dataframe_s2x = sort_satellites(prn_index_s2x, prn_values_s2x)
prn_dataframe_s2x['obs_type'] = 'S2X'

#S5X
reflector_s5x["prn"] = reflector_s5x["prn"].apply(clean_alt_list)
reflector_s5x["prn"] = reflector_s5x["prn"].apply(eval)

prn_index_s5x = to_1D(reflector_s5x["prn"]).value_counts().index
prn_values_s5x = to_1D(reflector_s5x["prn"]).value_counts().values

prn_dataframe_s5x = sort_satellites(prn_index_s5x, prn_values_s5x)
prn_dataframe_s5x['obs_type'] = 'S5X'

frames = [prn_dataframe_s1c, prn_dataframe_s2x, prn_dataframe_s5x]
combined_prn_dataframe = pd.concat(frames)
#------------------------------------------------Plot Data--------------------------------------------------------------
x_axis = np.arange(len(prn_dataframe_s1c['prn']))
legend_colors = {1:'tab:blue', 2:'tab:orange', 3:'tab:green'}

fig, ax = plt.subplots(figsize = (14,4))
ax.bar(x_axis, prn_dataframe_s1c['value'], width=1/4, color='tab:blue')
ax.bar(x_axis + 1/3, prn_dataframe_s2x['value'], width=1/4, color='tab:orange')
ax.bar(x_axis + 2/3, prn_dataframe_s5x['value'], width=1/4, color='tab:green')
ax.set_ylabel("Frequency", size = 12)
ax.set_xlabel("Satellite PRN", size = 12)
ax.set_title("3 Hour Window Reproducibility of Different GNSS Satellites", size = 14)
handles = [mpatches.Patch(color=legend_colors[i]) for i in legend_colors]
labels = ['S1C', 'S2X', 'S5X']
plt.legend(handles, labels)
plt.xticks(x_axis + 1/3, prn_dataframe_s1c['prn'], rotation=90)
plt.show()




