# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 11:21:29 2022

@author: fki1
"""

#### Script for Reading and Exporting files from Envi format to Raster. tif format
import numpy as np
# scipy.signal is for more complex signal processing manipulations
from scipy import signal
# is for manipulating files and filenames
# # for using functions related to ENVI, if you manage to install this, uncomment the following line, otherwise proceed without
import spectral.io.envi as envi      

# for file copying
import shutil
import rasterio as io
#import src
import sys
import os
from osgeo import gdal, gdalconst 
from osgeo.gdalconst import * 
import matplotlib.pyplot as plt
import fiona
#import rasterio
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from rasterio.mask import mask
import shapefile

#%%
def load_data(file_name, gdal_driver='GTiff'):
	'''
	Converts a GDAL compatable file into a numpy array and associated geodata.
	The rray is provided so you can run with your processing - the geodata consists of the geotransform and gdal dataset object
	If you're using an ENVI binary as input, this willr equire an associated .hdr file otherwise this will fail.
	This needs modifying if you're dealing with multiple bands.
	
	VARIABLES
	file_name : file name and path of your file
	
	RETURNS
	image array
	(geotransform, inDs)
	'''
	driver = gdal.GetDriverByName(gdal_driver) ## http://www.gdal.org/formats_list.html
	driver.Register()

	inDs = gdal.Open(file_name, GA_ReadOnly)

	if inDs is None:
		print("Couldn't open this file: %s" %(file_name))
		print('/nPerhaps you need an ENVI .hdr file? A quick way to do this is to just open the binary up in ENVI and one will be created for you.')
		sys.exit("Try again!")
	else:
		print("%s opened successfully" %file_name)
		
	# Extract some info form the inDs 		
	geotransform = inDs.GetGeoTransform()
		
	# Get the data as a numpy array
	band = inDs.GetRasterBand(1)
	cols = inDs.RasterXSize
	rows = inDs.RasterYSize
	image_array = band.ReadAsArray(0, 0, cols, rows)
	
	return image_array, (geotransform, inDs)


#%%
def array2raster(data_array, geodata, file_out, gdal_driver='GTiff'):
	'''
	Converts a numpy array to a specific geospatial output
	If you provide the geodata of the original input dataset, then the output array will match this exactly.
	If you've changed any extents/cell sizes, then you need to amend the geodata variable contents (see below)
	
	VARIABLES
	data_array = the numpy array of your data
	geodata = (geotransform, inDs) # this is a combined variable of components when you opened the dataset
				inDs = gdal.Open(file_name, GA_ReadOnly)
				geotransform = inDs.GetGeoTransform()
				see data2array()
	file_out = name of file to output to (directory must exist)
	gdal_driver = the gdal driver to use to write out the data (default is geotif) - see: http://www.gdal.org/formats_list.html

	RETURNS
	None
	'''

	if not os.path.exists(os.path.dirname(file_out)):
		print("Your output directory doesn't exist - please create it")
		print("No further processing will take place.")
	else:
		post=geodata[0][1]
		original_geotransform, inDs = geodata

		rows, cols = data_array.shape
		bands = 1

		# Set the gedal driver to use
		driver = gdal.GetDriverByName(gdal_driver) 
		driver.Register()

		# Creates a new raster data source
		outDs = driver.Create(file_out, cols, rows, bands, gdal.GDT_Float32)

		# Write metadata
		originX = original_geotransform[0]
		originY = original_geotransform[3]

		outDs.SetGeoTransform([originX, post, 0.0, originY, 0.0, -post])
		outDs.SetProjection(inDs.GetProjection())

		#Write raster datasets
		outBand = outDs.GetRasterBand(1)
		outBand.WriteArray(data_array)
			
		print("Output saved: %s" %file_out)



#%%
# In[53]:
# this is to run if it is the first time and files are not dorted yet
    
#path_pre = "D:\\WHH\\Frame_1\\3_Stack\\"
path = "D:\\WHH\\Frame_2\\4_Stack_Sorted\\"
#path_save = "D:\\WHH\\Frame_1\\5_Save\\"

list_dates = sorted(os.listdir(path))

num_acq = len(list_dates)


#%%
#converts tif files
#converts with projection issue
'''
for i in range(num_acq):
    file_name=("D:\\WHH\\Frame_2\\5_Save\\hdr\\c22\\{0}.img").format(list_dates[i]) 
    data, geodata = load_data(file_name)
    
    # Write it out as a geotiff
    file_out=("D:\\WHH\\Frame_2\\Converts\\clipped\\c22\\{0}.tif").format(list_dates[i])
    array2raster(data, geodata, file_out, gdal_driver='GTiff')

    
'''

#%% In[50]: 
#clipps tif files
shp_file_path = r"D:\\WHH\\Lake_Victoria\\Lake_shapefile\\lake_victoria.shp"
  
for i in range(num_acq):
    file_name=("D:\\WHH\\Frame_2\\Converts\\clipped\\c22\\{0}.tif").format(list_dates[i]) 
    
    output_raster_path = ("D:\\WHH\\Frame_2\\Converts\\tiff\\{0}.tif").format(list_dates[i])
    dst_crs = 'EPSG:4326'
    #with fiona.open(shp_file_path, "r") as shapefile:
     #   shapes = [feature["geometry"] for feature in shapefile]
    shapes = fiona.open(shp_file_path)
    

    with rasterio.open(file_name) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta
       
    out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": dst_crs})

    with rasterio.open(output_raster_path, "w", **out_meta) as dest:
        dest.write(out_image)
        
    del  file_name

#%%
'''
dst_crs = 'EPSG:4326'



inshp = "D:\\WHH\\Lake_Victoria\\Lake_shapefile\\lake_victoria_1.shp"
inRas = 'D:\\WHH\\Frame_2\\Converts\\clipped\\c22\\20170520.tif'
outRas = 'D:\\WHH\\Frame_2\\Converts\\tiff\\ClippedSmallRaster.tif'


Vector=Polygon(inshp)

#Vector=Vector[Vector['HYBAS_ID']==6060122060] # Subsetting to my AOI

with rasterio.open(inRas) as src:
    Vector=Vector.to_crs(dst_crs)
    # print(Vector.crs)
    out_image, out_transform= mask(src,Vector.geometry,crop=True)
    out_meta=src.meta.copy() # copy the metadata of the source DEM
    
out_meta.update({
    "driver":"Gtiff",
    "height":out_image.shape[1], # height starts with shape[1]
    "width":out_image.shape[2], # width starts with shape[2]
    "transform": dst_crs
})
              
with rasterio.open(outRas,'w',**out_meta) as dst:
    dst.write(out_image)
'''
