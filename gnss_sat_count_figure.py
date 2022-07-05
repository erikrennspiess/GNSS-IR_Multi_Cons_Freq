import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib import patches as mpatches


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

reflector_all = pd.read_csv('/home/OceanJasper/GNSS-R/altimetry/varying_window_sizes/frequencies/GNSS_frequencies/'
                            '6_hour_window_gps_galielo_glo_all.csv')

reflector_all["prn"] = reflector_all["prn"].apply(clean_alt_list)
reflector_all["prn"] = reflector_all["prn"].apply(eval)

prn_index = to_1D(reflector_all["prn"]).value_counts().index
prn_values = to_1D(reflector_all["prn"]).value_counts().values


prn_dataframe = pd.DataFrame({'prn': prn_index, 'value': prn_values})
prn_dataframe['sort_2'] = prn_dataframe['prn'].str.extract('(\d+)', expand=False).astype(int)
prn_dataframe.sort_values('sort_2', inplace=True, ascending=True)
prn_dataframe = prn_dataframe.drop('sort_2', axis=1)

letters = "GER"
d={i:letters.index(i) for i in letters}
sorted_prn = pd.DataFrame(sorted(prn_dataframe['prn'], key=lambda word:d[word[0]]), columns=['prn'])

sorted_dataframe = sorted_prn.join(prn_dataframe.set_index('prn'), on='prn')
cmap = plt.cm.get_cmap('tab20c')
colors = [cmap(5) if x[0] =='G' else cmap(6)
          if x[0] =='E' else cmap(7)
          for x in sorted_dataframe.prn]

legend_colors = {1:cmap(5), 2:cmap(6), 3:cmap(7)}

fig, ax = plt.subplots(figsize = (14,4)) #
ax.bar(sorted_dataframe['prn'], sorted_dataframe['value'], color=colors)
ax.set_ylabel("Frequency", size = 12)
ax.set_xlabel("Satellite PRN", size = 12)
ax.set_title("6 Hour Window Count of Different GNSS Satellites", size = 14)
handles = [mpatches.Patch(color=legend_colors[i]) for i in legend_colors]
labels = ['GPS', 'Galileo', 'GLONASS']
plt.legend(handles, labels)
plt.xticks(rotation=90)
plt.show()




