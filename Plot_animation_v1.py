# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:22:14 2023

@author: fki1
"""

import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio as io
from matplotlib.colors import ListedColormap, BoundaryNorm
#%%
# Load raster
#with rasterio.open("D:\\WHH\\Frame_2\\Converts\\clipped\\c22\\20170601_n.tif") as src:
    #raster = src.read(1)

# Load shapefile
shape = gpd.read_file("D:\\WHH\\Lake_Victoria\\Lake_shapefile\\lake_victoria.shp")

# Plot raster
colors = [ 'none', 'green',  ]
class_bins = [0, 1]
cmap = ListedColormap(colors)
norm = BoundaryNorm(class_bins, 
                    len(colors))
fig, ax = plt.subplots()
# Overlay shapefile

#ax.imshow(raster, cmap=cmap)

# Overlay shapefile

shape.plot(ax=ax, facecolor='none', edgecolor='red')

# Show plot
plt.show()
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


