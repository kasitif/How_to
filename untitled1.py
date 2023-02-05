# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 13:13:21 2023

@author: fki1
"""

# Import needed packages
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import geopandas as gpd
import rioxarray as rxr
from rasterio.plot import plotting_extent
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep

#%%

# Import fire boundary
shape = gpd.read_file("D:\\WHH\\Lake_Victoria\\Lake_shapefile\\lake_victoria.shp")

shape.crs
ep =shape.crs
###
data = rxr.open_rasterio(r"D:\\WHH\\Frame_2\\Converts\\clipped\\c22\\tif1.tif",  masked= True)
#data = rxr.open_rasterio(data_path, masked=True)
data_plotting_extent = plotting_extent(data)

# Getting the crs of the raster data
naip_crs = data.crs

# Transforming the fire boundary to the NAIP data crs
fire_bound_utmz13 = shape.to_crs(naip_crs)

# Opening the NAIP data
#naip_data = rxr.open_rasterio(naip_path, masked=True)

# Creating the plot extent object
naip_plot_extent = plotting_extent(data, 
                                   data.transform)

# Plot uncropped array
f, ax = plt.subplots()

ep.plot_bands(data.values,
            ax=ax,
            title="Fire boundary overlayed on top of uncropped NAIP data",
            extent=naip_plot_extent)  # Use plotting extent from DatasetReader object

fire_bound_utmz13.plot(ax=ax)

plt.show()