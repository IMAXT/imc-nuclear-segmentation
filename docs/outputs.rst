Data Description
================

Upon a successful run of the IMC pipeline, several data products are created inside the OUTPUT forlder (location and name of the OUTPUT folder are set by the user). These include several folders and files with their descriptions as follow:

1. CUBE_image (folder)
======================

*_CUBE.tiff : IMC image data cube in TIF format. Each layer in the cube is associated to an IMC channel. 
*_CUBE.txt  : The name of IMC channels and their orders as appear in the cube data file.

Description:

For each IMC run, the structure of the output data, need to be processed by the IMC pipeline, belongs to one of the following categories:

(a) Normal TIF files
--------------------
A collection of normal TIF files where the number of TIF files are the same as the number of IMC channels, and the name of each TIF file is the name of that channel.

(b) OME-TIF files
-----------------
These are produced if the output of an IMC run (raw data) are processed by the Data Packager. In this case, each image channel is of OME.TIF format and the channel names are recorded as meta data inside the image file.

Once the IMC pipeline starts, it first determines the structure of the input data files ('a' or 'b' above) and then produces a single TIF image data cube '*_CUBE.tiff' by merging all individual IMC image channels. In addition, the order of channels and their names are recorded in '*_CUBE.txt' file.


2. mask (folder)
================

*_mask.tif  : A 32-bit image mask of all detected cells. All pixels belong the same cell, have the same integer pixel values.

3. mask_each_cell (folder)
==========================

For each detected cell, a binary image mask is created and kept in this folder. For example, if there are 1000 detected cells, then you would see 1000 binary images in this folder. The index (or filename) of each binary image is the same as those in the final output catalog (inside the 'catalogue' folder).

3. reference (folder)
=====================

*_CUBE.jpg           : Image of the reference channel (nuclear channel) used for nuclear segmentation
flat_*_CUBE.tif      : Image of the reference channel (nuclear channel) after applying intensity correction (if requested)
draft_*_CUBE.tif     : Image of the reference channel (nuclear channel) with detected cells' centroids overlaid on image
draft_cnt_*_CUBE.tif : Image of the reference channel (nuclear channel) with estimated cells' contour lines overlaid on image 

Description:

The pipeline uses one of the nuclear channels (set by the user) to perform cell segmentation. In a few cases, it may happen that an intensity gradient appears across the nuclear channel. For instance, the middle area of the image may look much darker than the peripheri of the tissue or outer edges of the image. This may lower the performance of the segmentation. In order to account for such effect, the user can apply an intensity correction to the nuclear channel image by setting the 'aic_apply_intensity_correction: True' in the input configuration file. In that case, the 'flat_*_CUBE.tif' represents the reference channel after applying the intensity correction. If there is no intensity correction, 'flat_*_CUBE.tif' is basically the same as the reference/nuclear channel.

Draft Images ('draft_*_CUBE.tif' and 'draft_cnt_*_CUBE.tif'):
Each time the user runs the IMC pipeline, he/she can check the performance and quality of the detection by looking at 'draft_*_CUBE.tif' and 'draft_cnt_*_CUBE.tif' images. If not happy with the overall detections (e.g., overstiemated or understimated), the user can then adjust the performance of the segmentation by twiking the input parameters. 

4. catalogue
============

*_CUBE.csv  : Output catalog in CSV format
*_CUBE.fits : Output catalog in FITS format

Description:

Besides of other output products, output catalogues, contain most of the information retrieved from the analysis of an IMC run. The catalogue comes in two formats: 

(a) FITS (Flexible Image Transport System ), and
(b) CSV (Comma-Separated Values). 

It is up to users to decide which format they choose for their own analysis. 

There is one row for every detected cell in the output catalogue. There are also several feature columns associated to positional, shape, and intensity information on each segmented cell as follow:

Feature columns related to positional information
-------------------------------------------------
X, Y : Centroid of each cell assuming buttom-left corner of the image as reference i.e. (0,0).
X_image, Y_image : Centroid of each cell assuming top-left corner of the image as reference i.e. (0,0).  

Feature columns related to shape information (shape descriptors)
----------------------------------------------------------------
area_cnt : Cell contour area (pix^2). It is equal to the number of pixels within the detected contour area following segmentation.
area_minCircle : Area of the smallest circle (pix^2) that can be fit within the contour area. Therefore: area_minCircle <= area_cnt
area_ellipse : Area of an ellipse (pix^2), fitted to the estimated cell's contour (i.e. a set of 2D points.). 
ell_angle : The angle (degrees) at the center of the ellipse, measured from the x-axis anticlockwise. If 0 or 180, it is a horizontal ellipse. If 90, or 270, it is a vertical ellipse. 
ell_smaj : The smi-major axis (pixel) of the fitted ellipse
ell_smin : The smi-minor axis (pixel) of the fitted ellipse 
ell_e : Eccentricity of the fitted ellipse. A circle is an ellipse with an eccentricity of zero. 

Feature columns related to intensity information
------------------------------------------------

flux_XX : Mean (average) intensity, estimated from the IMC XX channel, associated with pixels belong to a cell's contour (including contour pixels)
f_buffer_XX :  Mean (average) intensity, estimated from the IMC XX channel, associated with pixels around a cell's contour (the width of the buffer area in pixels is set by the user)

Description:
The IMC pipeline uses the mask extracted for each cell, in order to estimate the average (mean) pixel intensities within (flux) and around (f_buffer) a cell, using all available IMC channels. Note that a few of these may belong to calibration channels and therefore should not be included in any analysis. You can find a list of all IMC channel names in the 'CUBE_image' folder. 


