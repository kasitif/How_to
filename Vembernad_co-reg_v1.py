 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 

@author: Armando Marino
"""
#%%
#####################################################################################
""" Functions """
#####################################################################################





  
#####################################################################################
""" Imports """
#####################################################################################
import sys
sys.path.insert(0, 'C:\\WHH\\Frame_3\\')

import glob
import subprocess
import os
#import flood_mapping_utils as utils
#import numpy as np
#import matplotlib.pyplot as plt
#from scipy import stats
#import zarr
#from skimage import morphology
#import cv2
import configparser
import time

#%%
#####################################################################################
""" Paths from config file"""
#####################################################################################
config_file_path = "D:\\WHH\\Frame_3\\config.txt"
parser = configparser.ConfigParser()
parser.read(config_file_path)
############################################ Paths for graph 1: raw input and output processed image
root = parser.get("WeedWatch_config", "root")                  
print(root)


#%% Takes approx. 3 minutes
#####################################################################################


# location/filename of Master images
master_path = glob.glob(root+'1_Processed_image\\'+'*dim')[0]
graph_path_2= parser.get("WeedWatch_config", "graph_path_2")   

# location/filename of images to process
input_files = os.listdir(root+"0_Raw_Image\\") 

for i in range(len(input_files)):
    Raw_image_path = root+'0_Raw_Image\\'+input_files[i]
    image_date_str = input_files[i].split('_')[5].split('T')[0]
    Processed_image_path = root+'1_Processed_image\\'+ image_date_str
    Img_to_coregister_path = Processed_image_path+'.dim'
    
    # Location/filename of images co-registered 
    stack_out_path = root+'3_Stack\\'+image_date_str 
    SNAP_version_2 = parser.get("WeedWatch_config", "SNAP_version_2") 

    
    print('Procesing acquisition ' + str(i+1) + ' of ' + str(len(input_files)))
    
    tic = time.time()
    subprocess.call([
        SNAP_version_2,
        graph_path_2,
        #f'-Pinput1={files_in}',
        f'-Pinput1={master_path}'+','+f'{Img_to_coregister_path}',
        f'-Poutput1={stack_out_path}'
        ])
    toc = time.time()
    print(' Procesing acquisition ' + str(i+1) + ' took ' + str((toc-tic)/60) + ' min.')
