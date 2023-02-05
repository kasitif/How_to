#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 12:19:41 2022

@author: cristian
"""
import numpy as np 
import spectral.io.envi as envi
import os

def read_config_file_snap(folder1):
    global header
    global datatype
    
    """
    This module open Sentinel 1 incoherent images 
    The header can be read from the ENVI .hdr
    """
    lib = envi.open(folder1 + '.hdr')
#    header =  np.array([lib.nrows, lib.ncols])
    Ncol=lib.ncols
    Nrow=lib.nrows
    header =  np.array([Nrow, Ncol])
    datatype = lib.dtype
    return(Nrow, Ncol)

def Open_C_diag_element(filename):  
    f = open(filename, 'rb')
    img = np.fromfile(f, dtype=datatype, sep="")
    img = img.reshape(header).astype('float32')
    return(img)

def extract_slv_image_name(path):
    """
    Given the path to the co-registered stack, obtain the name of the C11 slave image

    Parameters
    ----------
    path : path to co-registered stack

    Returns
    -------
    C11 slave image name

    """
    C11_all,C22_all,C12_real_all,C12_imag_all = [],[],[],[]   
    # Append all files for each image component
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path,i)) and 'C11' in i:
            C11_all.append(i)
        if os.path.isfile(os.path.join(path,i)) and 'C22' in i:
            C22_all.append(i)
        if os.path.isfile(os.path.join(path,i)) and 'C12_real' in i:
            C12_real_all.append(i)
        if os.path.isfile(os.path.join(path,i)) and 'C12_imag' in i:
            C12_imag_all.append(i)
    
    # Select only the images, not metadata files of each image
    C11_images,C22_images,C12_real_images,C12_imag_images = [], [], [], []
    for a,b,c,d in zip(C11_all,C22_all,C12_real_all,C12_imag_all):
        if '.img' in a: C11_images.append(a)
        if '.img' in b: C22_images.append(b)
        if '.img' in c: C12_real_images.append(c)        
        if '.img' in d: C12_imag_images.append(d) 
        
    
    # Select only the slaves
    C11_images_slv,C22_images_slv,C12_real_images_slv,C12_imag_images_slv = [], [], [], []
    for a,b,c,d in zip(C11_images,C22_images,C12_real_images,C12_imag_images):
        if 'slv' in a: C11_images_slv.append(a)
        if '.img' in b: C22_images_slv.append(b)
        if '.img' in c: C12_real_images_slv.append(c)        
        if '.img' in d: C12_imag_images_slv.append(d) 
        
    img_name=C11_images_slv[0].split('.')[0]
    print(img_name)
    return(img_name)