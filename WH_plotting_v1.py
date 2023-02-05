# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 23:14:14 2023

@author: fki1
"""

# Numpy is for Numerical manipulations
import numpy as np
# matplotlib.pyplot is for making plots
import matplotlib.pyplot as plt
# scipy.signal is for more complex signal processing manipulations
from scipy import signal
# is for manipulating files and filenames
import os
import xarray as xr
#import rioxarray
import spectral.io.envi as envi      
#import shutil
import rasterio as io
import geopandas as gpd
import time
from rasterio.transform import from_origin
import numpy as np
from scipy import signal

import sys
from rasterio import plot as rasterplot
from rasterio.plot import show
#import fiona
#import shapefile
#from osgeo import gdal, gdalconst, ogr, osr
#from osgeo.gdalconst import * 


#%%
#difff = xr.open_dataset("my.tif", engine="rasterio")
#D:\WHH\Frame_1\Coverts\WHLLDif_20181110.tif
diff = xr.open_rasterio(r"D:\\WHH\\Frame_2\\Converts\\clipped\\c22\\tif1.tif",  )


plt.figure(constrained_layout=True)
plt.title('Heat map for acquatic weeds')
#lake.plot(ax=ax, facecolor='w', edgecolor='k')
plt.imshow(diff[0,:,:], cmap = 'jet', vmin = 0, vmax = 1)
plt.colorbar()
plt.savefig('Heat_map_for_acquatic_weeds.png')


#%%
print(diff)
#%%
##read shapefile
#lake = shapefile.Reader(r"C:\\Users\\fki1\\Lake.shp")
##reading WH file
WH1 = io.open(r"D:\\WHH\\Frame_2\\Converts\\clipped\\c22\\tif1.tif",  masked= True)
WH2 = io.open(r"D:\\WHH\\Frame_2\\Converts\\clipped\\c22\\tif1.tif",  masked= True)
WH3 = io.open(r"D:\\WHH\\Frame_1\\Coverts\\clipped\\20170308.tif",  masked= True)
WH4 = io.open(r"D:\\WHH\\Frame_1\\Coverts\\clipped\\20170320.tif",  masked= True)
#WH5 = io.open(r"D:\WHH\Frame_1\Coverts\WH\LLDif_20181228.tif",  masked= True)
#WH6 = io.open(r"D:\WHH\Frame_1\Coverts\WH\LLDif_20190121.tif",  masked= True)
#fig, axs = plt.subplots(ncols=2, nrows=2)
#show(WH1)
#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout=True)
fig, ((axr, axg, ), ( axc,axd )) = plt.subplots(2,2, constrained_layout=True)
show((WH1, 1), ax=axr, cmap='Greens', title='10-11-2018', transform=None,)
show((WH2, 1), ax=axg, cmap='Greens', title='22-11-2018', )
show((WH3, 1), ax=axc, cmap='Greens', title='04-12-2018')
show((WH4, 1), ax=axd, cmap='Greens', title='16-12-2018')
#show((WH5, 1), ax=axd, cmap='Greens', title='28-12-2018')
#show((WH6, 1), ax=axe, cmap='Greens', title='21-01-2019')


plt.show()



