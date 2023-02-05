# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 11:04:56 2023

@author: kasit
"""

from rasterstats import zonal_stats, point_query
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import geopandas as gpd
import rioxarray as rxr
import rasterio as io
from rasterio.plot import plotting_extent
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.dates as mdates


#%%
'''
stats = []
cmap = { 'WH' : 1,'Clear':  0 }
zona_stats = zonal_stats("C:\\WHH\\Lake_shapefile\\lake_victoria1_2.shp", 'C:\\WHH\\tiff\\20170131.tif', categorical=True)
whh = zona_stats[0]
whh
wh =whh.get(1)
wh
date = 2010
def create_tuple(date, wh):
    date = date
    wh = wh
    return date , wh
stats = create_tuple(date, wh)
stats
'''
#%%
#gettin paths
path = "C:\\WHH\\tiff_combines\\"
list_dates = sorted(os.listdir(path))
num_acq = len(list_dates)

#define fucntion for 
#creating an empoty numoy array
arr = np.empty((0,3), int)

#loop throught the images ina folder whilke checking for the dates and the area of the WHH per image
for i in range(1,num_acq):
    name = ("C:\\WHH\\tiff_combines\\{0}").format(list_dates[i])
    
    #getting stats within teh shapefile
    zona_stats = zonal_stats("C:\\WHH\\Lake_shapefile\\lake_victoria1_2.shp", name, categorical=True)
    whh = zona_stats[0]
    wh =np.float32((whh.get(1))*0.000225) #multiplied by the area of one grid( approx 15 by 15 metres) in km2
    clear =int(whh.get(0))
    s = list_dates[i]
    #convert file names to date objects
    date = datetime(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]))
    arr = np.append(arr, np.array([[date,wh,clear]]), axis=0)
    print( 'Working on ' + str(name))
    del name, zona_stats

# saving outputs to CSV
df = pd.DataFrame(arr)
os.makedirs(path, exist_ok=True)  
df.to_csv('../combined_stat.csv')

#%%
# Plots of WH time series over a frame
#getting axis data
x  = arr[:, 0] #dates
y  = arr[:, 1] #WH coverage

labels = x

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=144))
plt.plot(x,y)
plt.xlabel('Date', fontsize= 18)
plt.ylabel('Water Hyacinth Coverage (Area Km2)', fontsize= 12)
plt.title('Area Coverage of WHH againts Time', fontsize= 18)
plt.grid(True)
plt.gcf().autofmt_xdate()

#%%
#PLOTS
'''
x  = arr[:, 0]
y  = arr[:, 1]
x
labels = x

plt.plot(x, y)
# You can specify a rotation for the tick labels in degrees or with keywords.
plt.xticks(x, labels, rotation='vertical')
# Pad margins so that markers don't get clipped by the axes
plt.margins(0.05)
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.xlabel('Date')
plt.ylabel('Water Hyacinth')
plt.title('Line plot of WHH over the years')
# Tweak spacing to prevent clipping of tick-labels
plt.grid(True)
plt.subplots_adjust(bottom=0.25)
plt.show()
'''
#

#



