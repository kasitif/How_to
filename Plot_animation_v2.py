# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:22:14 2023

@author: fki1
"""

# Import needed packages
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
import imageio.v2 as imageio

#%%
'''
# Load raster
# Get data and set working directory
data_path = 'C:\\WHH\\tiff\\'
# Load shapefile
shape = gpd.read_file("C:\\WHH\\Lake_shapefile\\lake_victoria1_2.shp")

csr= shape.crs

#
data = rxr.open_rasterio('C:\\WHH\\tiff\\20170131.tif', masked=True)
data = data.rio.reproject(csr)
#plot extend
data_plotting_extent = plotting_extent(data[0], 
                                   data.rio.transform())

# Getting the crs of the raster data
naip_crs = data.rio.crs

# Transforming the fire boundary to the NAIP data crs
shape_bound = shape.to_crs(naip_crs)


# See coordinates of plotting extent
data_plotting_extent


# Define color map
nbr_colors = ["None", "darkgreen"]
nbr_cmap = mpl.colors.ListedColormap(nbr_colors)


# Plot cropped data
f, ax = plt.subplots()

ep.plot_bands(data,
              ax=ax,
              cmap=nbr_cmap,
              extent=data_plotting_extent,
              cbar=False)

#ep.draw_legend(im_ax=im, classes=classes, titles=ndvi_cat_names)
shape.boundary.plot(ax=ax)
ax.set_title( "date", fontsize=14,)
ax.grid()
#ep.draw_legend(im)
plt.show()

'''
#%%

path = "C:\\WHH\\frame_1\\tiff_c22\\"
list_dates = sorted(os.listdir(path))
num_acq = len(list_dates)
shape = gpd.read_file("C:\\WHH\\Lake_shapefile\\lake_victoria1_2.shp")
# Define color map
nbr_colors = ["None", "darkgreen"]
nbr_cmap = mpl.colors.ListedColormap(nbr_colors)

pwd = "C:\\WHH\\frame_1\\"
  
for i in range(num_acq):
    name = ("C:\\WHH\\frame_1\\tiff_c22\\{0}").format(list_dates[i]) 
    print(name)
    data = rxr.open_rasterio(name)
    data_plotting_extent = plotting_extent(data[0], 
                                       data.rio.transform())
    
    #dates
    dates = list_dates[i]
    dates = str(dates[0:9])
    date = str(dates[0:4] + '-' + dates[4:6] +'-' + dates[6:9])
    
    # Plot cropped data
    f, ax = plt.subplots(figsize=(20, 16))

    ep.plot_bands(data,
                  ax=ax,
                  cmap=nbr_cmap,
                  extent=data_plotting_extent,
                  cbar=False)

    #ep.draw_legend(im_ax=im, classes=classes, titles=ndvi_cat_names)
    shape.boundary.plot(ax=ax)
    ax.set_title( date , fontsize=14) 
    ax.grid()
    #ep.draw_legend(im)
    
    plt.savefig( "C:\\WHH\\frame_1\\jpeg_plots\\" + date +'.jpg')
    plt.close()
#%%
#Animation with GIFF
#HAVE ALL JPEGS IN STHE SCRIPT FOLDER
filenames = []
y = "C:\\WHH\\combined_plots\\"
list_dates = sorted(os.listdir(y))
num_acq = len(list_dates)


# build gif
with imageio.get_writer('C:\\WHH\\frame_2\\mygif_2.gif', mode='I') as writer:
    for filename in list_dates:
        image = imageio.imread(filename)
        writer.append_data(image)
        



