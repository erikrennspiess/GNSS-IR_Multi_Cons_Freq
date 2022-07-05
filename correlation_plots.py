import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import gnss_reflector_tg_analysis.py



# making subplots
#fig3 =((ax2, ax3), (ax4, ax5), (ax6, ax7), (ax8, ax9)) = plt.subplots(ncols=2,nrows=4)
#fig3 = plt.figure()
#ax = fig3.add_subplot(111)
fig3, ax = plt.subplots(2,2)
ax2, ax3, ax4, ax5 = ax.flatten()


# set data with subplots and plot
#All freq
ax2.scatter(all_freq['estimated_reflector_height']+0.2, all_freq['tg_sea_level'],
                 c=all_freq['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'))

#L1 Band
ax3.scatter(s1c['estimated_reflector_height']+0.2, s1c['tg_sea_level'],
                 c=s1c['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'))
ax4.scatter(e1['estimated_reflector_height']+0.2, e1['tg_sea_level'],
                 c=e1['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'))
ax5.scatter(glo_s1c['estimated_reflector_height']+0.2, glo_s1c['tg_sea_level'],
                 c=glo_s1c['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'))

ax2.set_title("All Frequencies")
ax3.set_title("GPS-S1C")
ax4.set_title("GAL-E1")
ax5.set_title("GLO-S1C")

ax2.set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")
ax3.set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")
ax4.set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")
ax5.set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")

ax2.grid()
ax3.grid()
ax4.grid()
ax5.grid()

fig3.tight_layout()
norm = mpl.colors.Normalize(vmin=0, vmax=2)
fig3.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.get_cmap('RdYlBu_r')), ax=ax.ravel(), orientation='vertical',
                                   label='Estimated Reflector Height STD')
fig4, axb = plt.subplots(2,2)
ax6, ax7, ax8, ax9 = axb.flatten()

#L2 Band
ax6.scatter(s2x['estimated_reflector_height']+0.2, s2x['tg_sea_level'],
                 c=s2x['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'))
ax7.scatter(glo_s2c['estimated_reflector_height']+0.2, glo_s2c['tg_sea_level'],
                 c=glo_s2c['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'))

#L5 band
ax8.scatter(s5x['estimated_reflector_height']+0.2, s5x['tg_sea_level'],
                 c=s5x['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'))
ax9.scatter(e5['estimated_reflector_height']+0.2, e5['tg_sea_level'],
                 c=e5['no_of_sats'], cmap=plt.cm.get_cmap('RdYlBu_r'))


ax6.set_title("GPS-S2X")
ax7.set_title("GLO-S2C")
ax8.set_title("GPS-S5X")
ax9.set_title("GAL-E5a")

ax6.set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")
ax7.set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")
ax8.set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")
ax9.set(xlabel="Estimated Reflector Height", ylabel="Tide Gauge Sea Level")

ax6.grid()
ax7.grid()
ax8.grid()
ax9.grid()

# set spacing
fig4.tight_layout()
norm = mpl.colors.Normalize(vmin=0, vmax=2)
fig4.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.get_cmap('RdYlBu_r')), ax=axb.ravel(), orientation='vertical',
                                   label='Estimated Reflector Height STD')

plt.show()