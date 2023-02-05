
# coding: utf-8

# # PolSAR Time Series processing using python (on Jupyter Notebook)
# 
# ### Armando Marino, The University of Stirling, UK
# 
# In this practical, you will process a time series of either RADARSAT Constellation Mission **RCM** Imagery © (RADARSAT is an official mark of the Canadian Space Agency) or **SAOCOM** (Argentine Space Agency, SAOCOM® product - ©CONAE – 2020).  The RCM images were acquired in 2020 near Carman, Manitoba, Canada. They are a time series of quad-polarimetric C-band images. The SAOCOM images were acquired in 2020 and 2021 near Cordoba, Argentina. They are a time series of quad-polarimetric L-band images. All of these data can be used for training purposes only.
# 
# In the following practical, I have pre-processed the RCM and SAOCOM images to speed up the work and to make sure the files sizes are not too large. However, we are also providing the full images that you can pre-process on your own using SNAP. For RCM, I calibrated the data, made a subset, produced the coherency matrix, multilooked 2x1 to make the pixel more square, co-registered the stack, and geocoded the images. For SAOCOM, I calibrated the data, made a subset, produced the coherency matrix, co-registered the stack, and geocoded the images.
# 
# ## How to run the code
# In this practical we are using Jupyter Notebook. This makes the layout of the practical very clear and easy to follow, but the code itself could be run by any other Python editor (e.g. Spyder is my favourite).
# 
# In Jupyter Notebook, each of the blocks with a number *In []* on the left side is called a **Cell**. 
# To run a cell, you can click on it and then click *Ctrl+Enter* (*Shift+Return on a Mac OS*).
# Once you do this, an asterisk appears inside the brackets... this means that it is processing.
# 
# If you want to create a new cell, then click on the left hand side of the cell, this will go from green (Edit mode) to blue (Run mode). Then click "b" and a new sell will appear below. "d + d" is to delete a cell.  
# 
# The cells that do not have a number on the side are Markdown. These are comments from me to you.
# Inside a cell with code, the text that follows a "#" are comments, again Python will not run those, they are messages for us humans!
# 
# ## Level of complexity
# This practical is much more complex than the one you did with a single acquisition. Here, I would like to show you how you can apply what you learned in the previous practical to produce a more complex script. I understand this can be overwhelming if you have not coded before, and I suggest you make sure you understand the previous exercise before starting this one. Nevertheless, it is good for you to see how you can add complexity when producing a processing stack for polarimetric SAR data. 
# 

# ### <span style='color:Blue'> Bringing this to the next level:  
# Once you are confident you can run this in a Jupyter Notebook, download the file as .py and run the code in Spyder. </span>

#  # 1. PREPARATION MATTERS
# The first step is to import all the libraries that we will be using in our script.
# 
# In this practical I reduced the number of libraries used to a minimum. This makes the script easier to run on different machines, which may not have all the libraries installed. However, when you work on more complex scripts, you may want to take advantage of the large number of powerful libraries that are offered by the Python and SAR communities.   

# In[45]:


# system library
# import sys
# sys.path.insert(0, 'C:\\Programms\\Python\\Libraries\\')
# this library is used to tell Python where our functions or libraries are. Since we are working with a single script, 
# we will not use this library now but you may want to use it in the future. You need to make sure that the 
# folder is the one containing your user libraries. 

# Numpy is for Numerical manipulations
import numpy as np
# matplotlib.pyplot is for making plots
import matplotlib.pyplot as plt
# scipy.signal is for more complex signal processing manipulations
from scipy import signal
# is for manipulating files and filenames
import os
#import rioxarray as xr

# # for using functions related to ENVI, if you manage to install this, uncomment the following line, otherwise proceed without
import spectral.io.envi as envi      

# for file copying
import shutil
import rasterio as io
import geopandas as gpd
import time
from rasterio.transform import from_origin
import numpy as np
# scipy.signal is for more complex signal processing manipulations
from scipy import signal

# for file copying
import shutil
import sys

#from osgeo import gdal, gdalconst, ogr, osr
#from osgeo.gdalconst import * 

# Another useful library is GDAL (which we will not be using here because sometimes it is difficult to install on some machines). 

# You can even import your own libraries. To learn more about how to produce your own libraries, have a look at 
# some Python tutorials on the internet.


# # 2. DEFINING FUNCTIONS
# In the following Cells we will create all the functions we will use in this practical. When working on several projects it is strongly suggested that you write all these functions in an external Python library but for the purpose of this practical we will load the functions we need inline with the rest of the code.  
# 
# Most of the code in the following practical has been used in the previous practical. However, structuring the script with functions makes your code easier to read, expand, and export. I only included the main algorithms inside separate functions, but you may prefer to create even more functions when processing a single image...and you may even decide to write these functions inside a separate Python script (i.e., a **library**) where you have all your routine functions stored and maintained. This means that you will not even need to copy and paste the routine inside a new script, but you can just import the library with a command like:
# 
# import MyLibrary
# 
# where MyLibrary is your Python file.

# ## 2.1 Rearranging the filenames in the SNAP folder
# If you have used SNAP to produce a stack of images before, you know that SNAP follows a specific filename structure to create the files with the stacked images. However, the alphabetic order of these is arranging them in chronological order.  
# 
# These following 2 routines are used to *modify and rearrange* the filenames of each SNAP binary image 
# so that all the filenames are listed in a nice alphabetical and chronological order, which will allow us to easily create data cubes. 
# This is all done using native Python functions, treating the filenames as strings and manipulating them. There are other libraries that allow you to convert dates from one format to another and you may want to have a look at those in the future. However, I did not include those extra libraries here in order to keep the logical complexity as low as possible.
# 
# Also, some people may be tempted to change the names manually, which becomes a problem when 
# you have lots of filenames to modify. I definitely prefer Python to do this job for me!




# In[47]:


def convert_month_to_number(string):
    """
    This module converts the format of the month from a string to a number.
    INPUT
    string: the name of the month in letters
    OUTPUT 
    month: the name of the month in numbers
    """
    if string == 'Jan':
        month = '01'
    elif string == 'Feb':
        month = '02'
    if string == 'Mar':
        month = '03'
    elif string == 'Apr':
        month = '04'
    if string == 'May':
        month = '05'
    elif string == 'Jun':
        month = '06'
    if string == 'Jul':
        month = '07'
    elif string == 'Aug':
        month = '08'
    if string == 'Sep':
        month = '09'
    elif string == 'Oct':
        month = '10'
    if string == 'Nov':
        month = '11'
    elif string == 'Dec':
        month = '12'
        
    return month



#%% In[49]:
def Sort_Files_SNAP(path_pre, path):

#    path_pre = "E:\\Data\\WEED_WATCH\\3_Stack\\"
#    path = "E:\\Data\\WEED_WATCH\\4_Stack_sorted\\"
#    path_save = "E:\\Data\\WEED_WATCH\\4_Save\\"
    
    # getting all the names
    list_names_all = sorted(os.listdir(path_pre))
    num_acq = int(len(list_names_all)/2)
    list_dates = [None] * num_acq
    list_dir = [None] * num_acq
    
    # getting dir and dates and create new folders
    j = 0
    for i in range(2*num_acq):
        string = list_names_all[i]
        if string[-5:] == '.data':
            list_dates[j] = string[:-5]
            list_dir[j] = string[:-5]
            if not os.path.exists(path+string[:-5]):
                os.mkdir(path+string[:-5])
            j = j + 1
    
    # coping files and changing names
    for i in range(len(list_dir)):
        # copy subdirectory example  
    
        from_directory = path_pre + list_dir[i] + '.data'
        to_directory = path + list_dir[i]
        list_names_pre = sorted(os.listdir(from_directory))
        
        for j in range(0, len(list_names_pre)-1):
            src = from_directory + '\\' + list_names_pre[j] 
            drc = to_directory + '\\' + list_names_pre[j]
            shutil.copy2(src, drc)
    
     
        list_names = sorted(os.listdir(path+list_dir[i]))
        
        list_names_removed = [None] * len(list_names)
    
        k = 0
        for ii in range(len(list_names)):
            
            # it first removes all the filenames that are too long, keeping only the coregistered images
            if len(list_names[ii]) < 15:
                os.remove(path + list_names[i])
            
            elif 'mst' in list_names[ii]:
                os.remove(path + list_dir[i] + '\\' + list_names[ii])
                
            else:
                for iii in range(4):
                    char_slv = 'slv'+str(iii)+'_' 
                    if char_slv in list_names[ii]:
                        list_names_removed[k] = list_names[ii].replace(char_slv,'')
    
                        string = list_names_removed[k]
                        k = k + 1
                        extension = string[-4:]
                        string_date = string[-13:]
                        string_date = string_date[:-4]
                        month = convert_month_to_number(string_date[2:5])
                        string_date_new = string_date[-4:] + month + string_date[:2]
                        string_new  = string[:-13] + string_date_new + extension
                        if not os.path.exists(path + list_dir[i] + '\\' + string_new):
                            os.rename(path + list_dir[i] + '\\' + list_names[ii], path + list_dir[i] + '\\' + string_new)                    


    return list_dates


#%%
def Open_ENVI_Image(filename):
    """
    This module opens binary images 
    The parameters of the functions can be read in the ENVI header file .hdr
    col: number of columns in the image, if left empty takes the values directly from the header
    row: number of rows in the image, if left empty takes the values directly from the header
    dtype: data type, if left empty takes the values directly from the header 
    """
     # UNCOMMENT the following 3 lines if you manage to install Spectral 
    lib = envi.open(filename + '.hdr')
    header =  np.array([lib.ncols, lib.nrows])
    datatype = lib.dtype

#    # COMMENT the following 2 lines if you manage to install Spectral
#    header =  np.array([col, row])
#    datatype = dtype
#    
    # Opening the first image
    f = open(filename + '.img', 'rb')
    img = np.fromfile(f, dtype=datatype)
    
 
    img = img.reshape(header,order='F').astype(datatype)

    return(img)


# ## 2.3 Loading the full covariance matrix
# The next function is used to load the quad-pol coherency/covariance matrix for each of the acquisitions. You will recognise it is basically the same code we used for a single acquisition, just generalised using the input string "date". 

#%% In[50]:


def Load_One_Date_Dual(path, date):
    """
    This module opens the elements of a coherency matrix for a single acquisition.
    INPUT
    path: a string containing the path to the folder where the images are contained 
    date: a string containing the date of the acquisition we want to open
    OUTPUT 
    The elements of the coherency matrix stored as separate images
    """
    
    ############### Loading T11 ######################################
    # First we need to identify the name of the file for the HH image 
    fileC11 = "C11_" + date
    C11Full = Open_ENVI_Image(path + fileC11) 
    # Full stands for the "Entire image"
    # Notice I am calling a function by writing its name and passing the parameters separated by a comma.
    # Always make sure that the order of the parameters is consistent with your definition of the function done above. 
    
    ############### Loading T22 ############################
    fileC22 = "C22_" + date
    C22Full = Open_ENVI_Image(path + fileC22) 
    
    
    ############### Loading T12 ############################
    # The off diagonal terms are cross correlation and therefore they are complex
    # numbers. They are stored by SNAP as Real and Imaginary parts. Both are float numbers.
    fileC12_real = "C12_real_" + date
    C12Full_real = Open_ENVI_Image(path + fileC12_real) 
    fileC12_imag = "C12_imag_" + date
    C12Full_imag = Open_ENVI_Image(path + fileC12_imag) 
    
    # We can now put the Real and Imaginary parts together to form the complex number
    C12Full = C12Full_real + 1j*C12Full_imag
    # Since the Real and Imaginary part on their own are redundant, we can remove 
    # them from the RAM memory
    del C12Full_real, C12Full_imag
    
    return C11Full, C22Full, C12Full


# ## 2.4 Cloude-Pottier decomposition
# In this following module we want to apply the Cloude-Pottier decomposition (i.e. diagonalisation of the coherency matrix) to extract parameters for each date in the time series. You may notice that this is the same code we used before for one acquisition, so that we can repeat it N times over the entire time series.

# In[51]:
def Boxcar_Pol_filter_Dual(T11_pre, T22_pre, T12_pre, win):
#    win = [7, 7]
    win1 = np.int(win[0])
    win2 = np.int(win[1])
    
    # Create the kernel of the boxcar
    kernel  = np.ones((win1,win2),np.float32)/(win1*win2)
    
    # Filter the images using convolve2d
    T11 =  signal.convolve2d(T11_pre, kernel, 
                                  mode='same', boundary='fill', fillvalue=0)
    T22 =  signal.convolve2d(T22_pre, kernel, 
                                  mode='same', boundary='fill', fillvalue=0)
    T12 =  signal.convolve2d(T12_pre, kernel, 
                                  mode='same', boundary='fill', fillvalue=0)


    return T11, T22, T12



#%%

def Cloude_Pottier(T11, T22, T33, T12, T13, T23):
    """
    This module runs the Cloude-Pottier decomposition for a single acquisition. 
    NOTE, the images imported MUST be in Pauli basis.
    INPUT
    The elements of the coherency matrix     
    OUTPUT 
    Entropy
    Anisotropy
    Averaged Alpha angle
    """
    
    sizeI = np.shape(T11)            # evaluate the size of each image
    dim1 = sizeI[0]       
    dim2 = sizeI[1]   

    # We first initialise an empty 3x3 matrix  
    T = np.matrix(np.zeros([3,3],dtype=np.complex64)) 
    
    # Then we initialise the output of the decomposition. 
    lam1 = np.zeros([dim1,dim2])            # maximum eigenvalue
    lam2 = np.zeros([dim1,dim2])            # second eigenvalue
    lam3 = np.zeros([dim1,dim2])            # minimum eigenvalue
    alpha = np.zeros([dim1,dim2])           # mean alpha: characteristic angle
    alpha1 = np.zeros([dim1,dim2])          # dominant alpha
    alpha2 = np.zeros([dim1,dim2])          # second alpha
    alpha3 = np.zeros([dim1,dim2])          # weakest alpha
    beta = np.zeros([dim1,dim2])            # mean beta: related to the orientation angle
    beta1 = np.zeros([dim1,dim2])           # dominant beta
    beta2 = np.zeros([dim1,dim2])           # second beta
    beta3 = np.zeros([dim1,dim2])           # weakest beta
    H = np.zeros([dim1,dim2])               # Entropy
    Ani = np.zeros([dim1,dim2])             # Anisotropy
    
    # A rounding factor for regularising the calculations when we have zeros
    eps = 1e-9
    
            
    # To compute pixel by pixel we can use a for loop. This goes from the start to 
    # the end of the variable defined in "range()" and applies the same processing commands in each iteration.
    # Only the index pointing at the image pixel is changed at each iteration. 
    for m in range(0, np.int(dim1)-1):
        for n in range(0, np.int(dim2)-1):
            
            # We assign the values to the coherency matrix
            T[0,0] = T11[m,n]+eps
            # eps will regularise when we have a zero pixel. It will ensure the  
            # matrix to be full-rank and therefore invertible 
            T[0,1] = T12[m,n] 
            # We only need to regularise the elements of the diagonal 
            T[0,2] = T13[m,n]
            T[1,0] = np.conj(T12[m,n])
            T[1,1] = T22[m,n]+eps
            T[1,2] = T23[m,n]
            T[2,0] = np.conj(T13[m,n])
            T[2,1] = np.conj(T23[m,n])
            T[2,2] = T33[m,n]+eps
    
            # Diagonalise the matrix
            [d, v] = np.linalg.eigh(T)
            # eigh diagonalises Hermitian matrices. In the future, don't 
            # use it if your matrix is not Hermitian. 
            # d contains the eigenvalues as a vector
            # v contains the eigenvectors one beside the other, forming a matrix
            
            ind = np.argsort(d)     # we want to sort them from largest to smallest
            lam1[m,n] = d[ind[2]]
            lam2[m,n] = d[ind[1]]
            lam3[m,n] = d[ind[0]]
    
            # Evaluate the probabilities of each scattering mechanism
            P1 = lam1[m,n]/(lam1[m,n]+lam2[m,n]+lam3[m,n])
            P2 = lam2[m,n]/(lam1[m,n]+lam2[m,n]+lam3[m,n])
            P3 = lam3[m,n]/(lam1[m,n]+lam2[m,n]+lam3[m,n])
    
            # Evaluate the entropy
            H[m,n] = -(P1*np.log10(P1)/np.log10(3) + P2*np.log10(P2)/np.log10(3) + 
                     P3*np.log10(P3)/np.log10(3))
            
            # Evaluate the anisotropy
            Ani[m,n] = (lam2[m,n]-lam3[m,n])/(lam2[m,n]+lam3[m,n])
                            
            # Alpha angles
            alpha1[m,n] = np.arccos(np.abs(v[0,2]))
            alpha2[m,n] = np.arccos(np.abs(v[0,1]))
            alpha3[m,n] = np.arccos(np.abs(v[0,0]))
            # Mean alpha as coming from a Bernulli process
            alpha[m,n] = P1*alpha1[m,n] + P2*alpha2[m,n] + P3*alpha3[m,n]
            
            # Beta angle
            beta1[m,n] = np.arccos(np.abs(v[1,2])/np.sin(alpha1[m,n]))
            beta2[m,n] = np.arccos(np.abs(v[1,1])/np.sin(alpha2[m,n]))
            beta3[m,n] = np.arccos(np.abs(v[1,0])/np.sin(alpha3[m,n]))
            # Mean beta as coming from a Bernulli process
            beta[m,n] = P1*beta1[m,n] + P2*beta2[m,n] + P3*beta3[m,n]
    
         # it may be too much information to show the count-down as well
#        if np.remainder(m,10) == 0:
#            print(np.int(dim1)-m)
         
         
    return H, Ani, alpha 

#%%
########################################################################
def Opt_ChangeDet_dual(C11_a, C22_a, C12_a, C11_b, C22_b, C12_b):    
    
    dim = np.shape(C11_a) 
    dimr = dim[1] 
    dima = dim[0] 
        
    
    Ta = np.zeros((2,2), dtype=np.complex64) 
    Tb = np.zeros((2,2), dtype=np.complex64) 

#    pow1 = np.zeros((dima,dimr))
#    pow2 = np.zeros((dima,dimr))

    dif1 = np.zeros((dima,dimr))
    dif2 = np.zeros((dima,dimr))

#    wis = np.zeros((dima,dimr))

  
    for i in range(0, dima-1):
        for ii in range(0, dimr-1):             
            Ta[0,0] = C11_a[i,ii] + 1e-6     
            Ta[1,1] = C22_a[i,ii] + 1e-6                  
            Ta[0,1] = C12_a[i,ii]             
            Ta[1,0] = np.conj(Ta[0,1])             

            Tb[0,0] = C11_b[i,ii] + 1e-6    
            Tb[1,1] = C22_b[i,ii] + 1e-6                  
            Tb[0,1] = C12_b[i,ii]             
            Tb[1,0] = np.conj(Tb[0,1])             

#            invTb = ln.inv(Tb)
#            A = np.matmul(invTb, Ta)
            B = Ta - Tb
            
#            # Power ratio
#            [d1, v1] = ln.eig(A)
#            
#            pow1[i,ii] = np.abs(np.max(d1))        
#            pow2[i,ii] = np.abs(1./np.min(d1))
#            
            # Power difference
            [d2, v2] = np.linalg.eigh(B)
            
            dif1[i,ii] = np.max(d2)        
            dif2[i,ii] = np.min(d2)
            
#            # Wishart
#            wis[i,ii] = np.trace(A)
            
            
#        print(dima-i)
    


    return dif1, dif2


#%% In[53]:
# this is to run if it is the first time and files are not dorted yet
    
path_pre = "D:\\WHH\\Frame_3\\3_Stack\\"
path = "D:\\WHH\\Frame_3\\4_Stack_Sorted\\"
path_save = "D:\\WHH\\Frame_3\\5_Save\\"

##############################################
#list_dates = Sort_Files_SNAP(path_pre, path)
##############################################

list_dates = sorted(os.listdir(path))

num_acq = len(list_dates)



#%% Getting size of images 

filename = path + list_dates[0] + "\\C11_" + list_dates[0]

lib = envi.open(filename + '.hdr')
header =  np.array([lib.ncols, lib.nrows])
datatype = lib.dtype


C11_cube = np.zeros([header[0],header[1],num_acq]) 
C22_cube = np.zeros([header[0],header[1],num_acq]) 
C12_cube = np.zeros([header[0],header[1],num_acq]) 
#lam1_cube = np.zeros([header[0],header[1],num_acq]) 
lam2_cube = np.zeros([header[0],header[1],num_acq])

 
#%% In[56]:


# we need a for loop to run through all the acquisitions in the time series
for i in range(num_acq):
    
    # the following command produces a print out that allows us to know how much is missing. 
    print('Pre-Processing date ' + str(i+1) + '....... ' + str(num_acq-i-1) + ' dates left.' )
    
    # We can load the images using the function we defined previously
    path_img = path+list_dates[i] + '\\'
    [C11Full, C22Full, C12Full] = Load_One_Date_Dual(path_img, list_dates[i])

#    # we want to take a crop of the full image to avoid issues with the limited RAM    
#    C11_pre = C11Full[dr1:dr2, da1:da2]           
#    C22_pre = C22Full[dr1:dr2, da1:da2]
#    C12_pre = C12Full[dr1:dr2, da1:da2]            

    C11_pre = C11Full
    C22_pre = C22Full
    C12_pre = C12Full
    del C11Full, C22Full, C12Full

    win = [7,7]
    [C11, C22, C12] = Boxcar_Pol_filter_Dual(C11_pre, C22_pre, C12_pre, win)
    del C11_pre, C22_pre, C12_pre
    
  
    C11_cube[:,:,i] = C11
    C22_cube[:,:,i] = C22
    C12_cube[:,:,i] = C12
    

C11_mean = np.nanmean(C11_cube, axis = 2)
C22_mean = np.nanmean(C22_cube, axis = 2)
C12_mean = np.nanmean(C12_cube, axis = 2)
### 
ref_ind = list_dates.index("20171003")
#save cubes

np.save('D:\\WHH\\Frame_3\\5_Save\\C11_cube', C11_cube)
np.save('D:\\WHH\\Frame_3\\5_Save\\C22_cube', C22_cube)
np.save('D:\\WHH\\Frame_3\\5_Save\\C12_cube', C12_cube)

C11_ref = C11_cube[:,:,ref_ind]
C22_ref = C22_cube[:,:,ref_ind]
C12_ref = C12_cube[:,:,ref_ind]


#%% In[57]:

fact = 2
'''
plt.figure()
plt.title('Mean image VV')
plt.imshow(C11_mean, cmap = 'gray', vmin = 0, vmax = fact*np.nanmean(C11_mean))

plt.figure()
plt.title('Mean image VH')
plt.imshow(C22_mean, cmap = 'gray', vmin = 0, vmax = fact*np.nanmean(C22_mean))

plt.figure()
plt.title('Mean image magnitude of C12')
plt.imshow(np.abs(C12_mean), cmap = 'gray', vmin = 0, vmax = fact*np.nanmean(np.abs(C12_mean)))
'''



#%% In[58]:
# Applying the chnage detector
for i in range(num_acq):
#for i in range(2):

    tic = time.time()
    # the following command produces a print out that allows us to know how many images are left. 
    print('Detector processing date ' + str(i+1) + '....... ' + str(num_acq-i-1) + ' dates left.' )

    [dif1, dif2] = Opt_ChangeDet_dual(C11_cube[:,:,i], C22_cube[:,:,i], C12_cube[:,:,i], 
                                      C11_ref,         C22_ref,         C12_ref)

    #lam1_cube[:,:,i] = dif1
    lam2_cube[:,:,i] = dif2
    toc = time.time()
    print(' Procesing acquisition ' + str(i) + ' took ' + str((toc-tic)/60) + ' min.')



#np.save('D:\\WHH\\Frame_3\\5_Save\\lam1_cube', lam1_cube)

np.save('D:\\WHH\\Frame_3\\5_Save\\lam2_cube', lam2_cube)
#%%
'''
plt.figure()
plt.title('Change map at date' + list_dates[i])
plt.imshow(lam1_cube[:,:,i], cmap = 'gray', 
           vmin = 0, vmax = 10*np.nanmean(lam1_cube[:,:,i]))
'''
plt.figure()
plt.title('Change map at date' + list_dates[i])
plt.imshow(lam2_cube[:,:,i], cmap = 'gray', 
           vmin = 0, vmax = 10*np.nanmean(lam2_cube[:,:,i]))
# setting the threshold
thr = 0.05
det_cube = np.zeros([header[0],header[1],num_acq])
#det_cube[lam1_cube > thr] = 1

# setting the threshold
thr2 = 0.005
det_cube2 = np.zeros([header[0],header[1],num_acq])
det_cube2[lam2_cube > thr2] = 1

# evaluate the heat map
dif_sum = np.nanmean(det_cube, axis = 2)
dif_sum2 = np.nanmean(det_cube2, axis = 2)

plt.figure()
plt.title('Heat map for acquatic weeds')
plt.imshow(dif_sum, cmap = 'jet', vmin = 0, vmax = 1)
plt.colorbar()

plt.figure()
plt.title('Heat map for acquatic weeds')
plt.imshow(dif_sum2, cmap = 'jet', vmin = 0, vmax = 1)
plt.colorbar()



#%% Preparing for saving
# Rotating images

C11_mean_rot = np.transpose(C11)
C22_mean_rot = np.transpose(C22)
C12_mean_rot = np.transpose(C12)
dif_sum_rot = np.transpose(dif_sum)
dif_sum_rot2 = np.transpose(dif_sum2)


# visulise them
fact = 2

plt.figure()
plt.title('Mean image VV')
plt.imshow(C11_mean_rot, cmap = 'gray', vmin = 0, vmax = fact*np.nanmean(C11_mean))
plt.savefig('Mean_image_VV.png')

plt.figure()
plt.title('Mean image VH')
plt.imshow(C22_mean_rot, cmap = 'gray', vmin = 0, vmax = fact*np.nanmean(C22_mean))
plt.savefig('Mean_image_VH.png')

plt.figure()
plt.title('Mean image magnitude of C12')
plt.imshow(np.abs(C12_mean_rot), cmap = 'gray', vmin = 0, vmax = fact*np.nanmean(np.abs(C12_mean)))
plt.savefig('Mean_image_magnitude_of_C12.png')

plt.figure()
plt.title('Heat map for acquatic weeds')
plt.imshow(dif_sum_rot, cmap = 'jet', vmin = 0, vmax = 1)
plt.colorbar()
plt.savefig('Heat_map_for_acquatic_weeds.png')

plt.figure()
plt.title('Heat map for acquatic weeds 2')
plt.imshow(dif_sum_rot2, cmap = 'jet', vmin = 0, vmax = 1)
plt.colorbar()
plt.savefig('Heat_map_for_acquatic_weeds.png')

#%%
# saving one image as envi file

envi.save_image('D:\\WHH\\Frame_3\\5_Save\\C11_mean_rot.hdr', C11_mean_rot, 
                dtype=np.float32, metadata=lib.metadata,force=True)
envi.save_image('D:\\WHH\\Frame_3\\5_Save\\C22_mean_rot.hdr', C22_mean_rot, 
                dtype=np.float32, metadata=lib.metadata,force=True)
envi.save_image('D:\\WHH\\Frame_3\\5_Save\\C12_mean_rot.hdr', C12_mean_rot, 
                dtype=np.float32, metadata=lib.metadata,force=True)
envi.save_image('D:\\WHH\\Frame_3\\5_Save\\dif_sum_rot.hdr', dif_sum_rot, 
                dtype=np.float32, metadata=lib.metadata,force=True)

#np.save('D:\\WHH\\Frame_3\\5_Save\\lam1_cube', lam1_cube)

np.save('D:\\WHH\\Frame_3\\5_Save\\lam2_cube', lam2_cube)

#%% 
#Applying and saving indvidual dates
'''
for i in range(num_acq):
    file_date = lam2_cube[:,:,i]
    # setting the threshold
    thr = 0.006
    det_cube = np.zeros([header[0],header[1],i])
    det_cube[file_date > thr] = 1
    wh_date = np.nanmean(det_cube, axis = 2)
    wh_det = np.transpose(wh_date)

    #
    file = envi.save_image(('D:\\WHH\\Frame_3\\5_Save\\hdr\\{0}.hdr').format(list_dates[i]), wh_det, 
                    dtype=np.float32, metadata=lib.metadata,force=True)
    
    print('Saved  ' + '{0}'.format(list_dates[i]))
    del file
    #file_date = lam2_cube[:,:,i]
    # setting the threshold

 
 ''' 

#%%
# reading the saved cube
lam2_cube = np.load('D:\\WHH\\Frame_3\\5_Save\\lam2_cube.npy')
#lam1_cube = np.load('D:\\WHH\\Frame_3\\5_Save\\lam1_cube.npy')
# this is to run if it is the first time and files are not dorted yet

   
path_pre = "D:\\WHH\\Frame_3\\3_Stack\\"
path = "D:\\WHH\\Frame_3\\4_Stack_Sorted\\"
path_save = "D:\\WHH\\Frame_3\\5_Save\\"
#get list of dates
list_dates = sorted(os.listdir(path))

#get number of files
num_acq = len(list_dates)

#gettign headers
#%% Getting size of images 

filename = path + list_dates[0] + "\\C11_" + list_dates[0]
lib = envi.open(filename + '.hdr')
header =  np.array([lib.ncols, lib.nrows])
datatype = lib.dtype

#applying the change detector for all dates in Lam2_cube
for i in range(num_acq):
    file_date = lam2_cube[:,:,i]
    # setting the threshold
    thr = 0.006
    det_cube = np.zeros([header[0],header[1],i])
    det_cube[file_date > thr] = 1
    wh_date = np.nanmean(det_cube, axis = 2)
    wh_det = np.transpose(wh_date)

    #
    file = envi.save_image(('D:\\WHH\\Frame_3\\5_Save\\hdr\\c22\\{0}.hdr').format(list_dates[i]), wh_det, 
                    dtype=np.float32, metadata=lib.metadata,force=True)
    
    print('Saved  '  + '{0}'.format(list_dates[i]))
    del file, det_cube, wh_date, wh_det, file_date
# promp when done
print('Done!')   
#%%
'''
#detecting lam1
filename = path + list_dates[0] + "\\C11_" + list_dates[0]
lib = envi.open(filename + '.hdr')
header =  np.array([lib.ncols, lib.nrows])
datatype = lib.dtype

#applying the change detector for all dates in Lam1_cube
for i in range(num_acq):
    file_date = lam1_cube[:,:,i]
    # setting the threshold
    thr = 0.05
    det_cube = np.zeros([header[0],header[1],i])
    det_cube[file_date > thr] = 1
    wh_date = np.nanmean(det_cube, axis = 2)
    wh_det = np.transpose(wh_date)

    #
    file = envi.save_image(('D:\\WHH\\Frame_3\\5_Save\\hdr\\c11\\{0}.hdr').format(list_dates[i]), wh_det, 
                    dtype=np.float32, metadata=lib.metadata,force=True)
    
    print('Saved  '  + '{0}'.format(list_dates[i]))
    del file, det_cube, wh_date, wh_det, file_date
# promp when done
print('Done!')     
'''
#%% 
'''
# reading the saved cube
lamb1_cube = np.load('D:\\WHH\\Frame_1\\5_Save\\lam1_cube.npy')

# this is to run if it is the first time and files are not dorted yet
    
path_pre = "D:\\WHH\\Frame_1\\3_Stack\\"
path = "D:\\WHH\\Frame_1\\4_Stack_Sorted\\"
path_save = "D:\\WHH\\Frame_1\\5_Save\\"



list_dates = sorted(os.listdir(path))

num_acq = len(list_dates[0:29])

filename = path + list_dates[0] + "\\C11_" + list_dates[0]
lib = envi.open(filename + '.hdr')
header =  np.array([lib.ncols, lib.nrows])
datatype = lib.dtype
ref_ind = list_dates.index("20171209")

#slicing per years
lamb1_cube_2017 = lamb1_cube[:,:, 0:29]
lamb1_cube_2018 = lamb1_cube[:,:, 29:60]
lamb1_cube_2019 = lamb1_cube[:,:, 60:89]
lamb1_cube_2020 = lamb1_cube[:,:, 89:120]
lamb1_cube_2021 = lamb1_cube[:,:, 120:150]
lamb1_cube_2022 = lamb1_cube[:,:, 150:]

#detect annual Heatmaps
# setting the threshold
thr = 0.05
det_cube = np.zeros([header[0],header[1],num_acq])
det_cube[lamb1_cube_2018 > thr] = 1

# evaluate the heat map
dif_sum = np.nanmean(det_cube, axis = 2)
dif_sum_rot = np.transpose(dif_sum)


plt.figure()
plt.title('Heat map for acquatic weeds 2017')
plt.imshow(dif_sum_rot, cmap = 'jet', vmin = 0.01, vmax = 0.4)
plt.colorbar()

#envi.save_image('D:\\WHH\\Frame_1\\5_Save\\2017\\dif_sum_rot_2017.hdr', dif_sum_rot, dtype=np.float32, metadata=lib.metadata,force=True)

'''







#%%
#
##%%
## RGB
## if you could not save the Pauli elements please change the following line with sizeI = np.shape(T11) and it 
## will plot the last image of the stack 
#sizeI = np.shape(C11_cube)            # evaluate the size of each image
#dim1 = sizeI[0]       
#dim2 = sizeI[1]   
#iRGB = np.zeros([dim1, dim2, 3])    # Create an empty 3D array (full of zeros) 
#
#for i in range(num_acq):
#
#    # we assign to the generic arrays T11, T22, and T33, the specific elements of the data cube  
#    C11 = C11_cube[:,:,i]
#    C22 = C22_cube[:,:,i]
#    
#    iRGB = np.zeros([dim1, dim2, 3])  
#    
#    fact = 1.5       # try different values for this
#    iRGB[:,:,0] = C11/(C11.mean()*fact)
#    iRGB[:,:,1] = np.abs(C12/(np.abs(C11).mean()*fact))    
#    iRGB[:,:,2] = C22/(C11.mean()*fact)
#    iRGB[np.abs(iRGB) > 1] = 1
#
#    # by leaving the argument in parenthesis empty, we produce a new image
#    fig = plt.figure()     
#    plt.title('After BOXCAR FILTER. ' + list_dates[i])     
#    plt.imshow(iRGB)
#
#
#
#
## In[41]:
#
#
#
## as done before, we want to show the parameters for each image.
## Note, you could create a routine that can create a figure with 4 connected sub-figures 
#for i in range(num_acq):
#    
#    # We could compare the Alpha angles
#    fig, [(ax1, ax2), (ay1, ay2)] = plt.subplots(2, 2, sharex=True, sharey=True)
#    ax1.set_title('Alpha for date ' + list_date[i])
#    cax1 = ax1.imshow(alpha_cube[:,:,i], cmap = 'jet', vmin=0, vmax=np.pi/2)   # cmap sets the colour map for the image
#    plt.colorbar(cax1, ax=ax1)
#    ax2.set_title('Entropy for date ' + list_date[i])
#    cax2 = ax2.imshow(entropy_cube[:,:,i], cmap = 'gray', vmin=0, vmax=1)
#    plt.colorbar(cax2, ax=ax2)
#    ay1.set_title('Span for date ' + list_date[i])
#    cax3 = ay1.imshow(span_cube[:,:,i], cmap = 'gray', vmin=0, vmax=2.5*np.nanmean(span_cube[:,:,i]))
#    plt.colorbar(cax3, ax=ay1)
#    ay2.set_title('Anisotropy for date ' + list_date[i])
#    cax4 = ay2.imshow(anisotropy_cube[:,:,i], cmap = 'gray', vmin=0, vmax=1)
#    plt.colorbar(cax4, ax=ay2)
#
#
#
## # 5 EXTRACT INFO FOR A FIELD OF INTEREST
## When you do a time series analysis you often want to look at the trends of pixels or group of pixels. In agricultural studies we are often interested in characterising all pixels inside a parcel more than single pixels. This is especially true when we want to create models for crop trends.
## In the following section we select pixels inside a parcel using a rectangle (as you did previously to evaluate statistics). Ideally, you will have a mask identifying the specific field boundaries. Also it is easier to do this using geotiff and shapefiles more than binary files (using GDAL and RASTERIO). However, the complexity of using GDAL is outside the purpose of this practical and will divert our attention away from the polarimetric analysis. If you are interested in producing automatic processing stacks, my suggestion is that you use tools like GDAL, RASTERIO, ZARRAY.
## 
#
## In[42]:
#
#
## first let's select a specific field 
#if sensor == 'RCM':
#    r_start = 310
#    r_end   = 400
#    a_start = 310
#    a_end   = 400
#
#elif sensor == 'SAOCOM':    
#    r_start = 450
#    r_end   = 500
#    a_start = 200
#    a_end   = 300
#
## we need to create the containers for the time trends
#Crop1_entropy_time = np.zeros(num_acq)
#Crop1_alpha_time = np.zeros(num_acq)
#Crop1_anisotropy_time = np.zeros(num_acq)
#Crop1_span_time = np.zeros(num_acq)
#
## we can see these contained one by one
#for i in range(num_acq):
#    Crop1_entropy_time[i] = np.nanmean(entropy_cube[a_start:a_end,r_start:r_end,i])
#    # notice I am using nanmean here and not mean. This is because mean will have problems if you have Not A Number (NaN) 
#    # in the data. Although we made sure there are no NaN's here :) I wanted to show you this function in case you 
#    # will ned to use it in the future
#    Crop1_anisotropy_time[i] = np.nanmean(alpha_cube[a_start:a_end,r_start:r_end,i])
#    Crop1_alpha_time[i]      = np.nanmean(alpha_cube[a_start:a_end,r_start:r_end,i])
#    Crop1_span_time[i]       = np.nanmean(span_cube[a_start:a_end,r_start:r_end,i])
#
#
## ## 5.1 Visaulise the trends
#
## In[43]:
#
#
#plt.figure()
#plt.title('Entropy for time series')
#plt.plot(list_date, Crop1_entropy_time)
#plt.ylim(ymax = 1, ymin = 0)
#plt.xticks(rotation=45)
#
#plt.figure()
#plt.title('Anisotropy for time series')
#plt.plot(list_date, Crop1_anisotropy_time)
#plt.ylim(ymax = 1, ymin = 0)
#plt.xticks(rotation=45)
#
#plt.figure()
#plt.title('Alpha for time series')
#plt.plot(list_date, Crop1_alpha_time)
#plt.ylim(ymax = np.pi/2, ymin = 0)
#plt.xticks(rotation=45)
#
#plt.figure()
#plt.title('Span for time series')
#plt.plot(list_date, Crop1_span_time)
#plt.xticks(rotation=45)
#
#
## ### <span style='color:Blue'> Analyse SAOCOM data:  
## For SAOCOM, why do fields increase their entropy and their alpha angle?
## This depends on the phenological stages of plants. 
## 
## In December, plants are close to seeding, the plants are small and the surface scattering is stronger especially in L-band (a longer wavelength than C-band). Surface scattering has a lower entropy and alpha angle. In March, plants are close to harvest, the plants are larger and the volume scattering is much stronger, returning a larger entropy and average alpha angle.  
#
## ### <span style='color:Blue'> Analyse RCM data:  
## For RCM, why does the field under analysis has this behaviour in terms of entropy and alpha angle?
## 
## Every field may behave differently, the one I am analysing here has a decrease in entropy and a small decrease in alpha angle. This is an interesting case since the image looks bluish, but the average alpha is still rather high. This is mostly due to the fact that the entropy is still relatively high and when we produce the average this pushes the averaged alpha to a higher value. This can be due to plants getting drier, which makes surface scattering underneath more visible (this reduces the entropy, but there is still a volume component that increases the value of alpha). Also, since the span is going down, it is possible that the surface scattering is mixed with thermal noise, increasing the entropy and therefore the averaged alpha.
## 
#
## ### <span style='color:Blue'> Explore:  
## Can you find some fields where this is not happening? What is your hypothesis there, what is going on? 
## 
#
## # 6 SAVE THE CUBES
## It is good practice to save the result of the cubes so that in the future you will not need to start from scratch to generate all the cubes again. 
## 
## There are many ways to save/store results using Python and for consistency you may want to use ENVI. I leave this to you as an exercise to try. Here I am showing you another data format native to Python and very easy to use (numpy format). 
#
## In[44]:
#
#
#filename_entropy = "Entropy_Cube_subset1"    
#filename_anisotropy = "Anisotropy_Cube_subset1"    
#filename_alpha = "Alpha_Cube_subset1"    
#filename_span = "Span_Cube_subset1"    
#
#np.save(path_save+filename_entropy, entropy_cube)
#np.save(path_save+filename_anisotropy, anisotropy_cube)
#np.save(path_save+filename_alpha, alpha_cube)
#np.save(path_save+filename_span, span_cube)

