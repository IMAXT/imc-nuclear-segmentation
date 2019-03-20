# import numpy as np
# from scipy.misc import bytescale
# import cv2



# i/o - segmentation
import cv2
import numpy as np
import os, time, sys, random, glob , ntpath
import imc_packages as imcutil
from astropy.table import Table
from PIL import Image
import yaml

# watershed segmentation
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.misc import bytescale    # convert 16bit -> 8bit (deprecated in scipy 1.2.0)
from scipy.ndimage.filters import gaussian_filter

# paralellization
from distributed import LocalCluster, Client, as_completed
from dask import delayed


'''
########################################### I M C    F U N C T I O N S    S T A R T  ##############################################
'''




def get_cnt_mask(cluster_index, sp_arr, labels_shape):
    # time.sleep(1)
    cnt_y_index, cnt_x_index = np.unravel_index(sp_arr.indices[sp_arr.data == cluster_index], labels_shape)
    
    cnt_x_min, cnt_x_max, cnt_y_min, cnt_y_max = np.min(cnt_x_index), np.max(cnt_x_index), np.min(cnt_y_index), np.max(cnt_y_index)
    cnt_topLeft_P = (cnt_x_min, cnt_y_min)
    cnt_img_h , cnt_img_w = cnt_y_max - cnt_y_min + 1 , cnt_x_max - cnt_x_min + 1
    cnt_img_shape = (cnt_img_h, cnt_img_w)
    cnt_mask_x_index = cnt_x_index - cnt_x_min
    cnt_mask_y_index = cnt_y_index - cnt_y_min
    cnt_mask_xy_index = (cnt_mask_y_index , cnt_mask_x_index)
    cnt_mask = np.zeros(cnt_img_shape, dtype="uint8")
    cnt_mask[cnt_mask_xy_index] = 1    
    
    
    return cnt_mask, cnt_topLeft_P


def get_contour_in_mask(cnt_mask, cnt_topLeft_P):
    
    # detect contours in the cnt_mask and grab the largest one
    cnts = cv2.findContours(cnt_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    c = max(cnts, key = cv2.contourArea)
    c =  c.reshape(c.shape[0],c.shape[2])
    
    # apply offset to account for the correct location of the cell in the image
    c[:,0] += cnt_topLeft_P[0]
    c[:,1] += cnt_topLeft_P[1]
    
    return c

def get_feature_table(n_valid_cnt = 0):

    # initialize number of elements

    # positional parameters
    feature_X_topcat , feature_Y_topcat , feature_X_image, feature_Y_image = [np.zeros(shape = n_valid_cnt, dtype = np.uint32) for _ in range(4)]
    
    # morphological / geometrical parameters
    feature_area_cnt , feature_area_mincircle , feature_area_ellipse = \
                            [np.zeros(shape = n_valid_cnt, dtype = np.uint32) for _ in range(3)]
    
    # fitted ellipse                                    
    feature_ell_angle, feature_ell_smaj  , feature_ell_smin  , feature_ell_e = \
                            [np.zeros(shape = n_valid_cnt, dtype = np.float32) for _ in range(4)]
    
    # intensity parameters: There are 'ch' number of channels for each image
    ch = 40 # for the time being, set it to 40. Alternatively, we can read it from the number of available channels in the input data cube
    flux = np.zeros((ch, n_valid_cnt), dtype = np.float32)
    f_buffer = np.zeros((ch, n_valid_cnt), dtype = np.float32)
    
    #  ------------------------------------------------------------
    # |                 creating binary table                      |
    #  ------------------------------------------------------------
    
    t = Table(
        [
            feature_X_topcat, feature_Y_topcat, feature_X_image, feature_X_image,               # positional-related                                     
            
            feature_area_cnt, feature_area_mincircle, feature_area_ellipse,                     # size-related
            
    
            
            feature_ell_angle, feature_ell_smaj, feature_ell_smin, feature_ell_e,               # fitted ellipse
    
            
            flux[0] ,flux[1] ,flux[2] ,flux[3] ,flux[4] ,flux[5] ,flux[6] ,flux[7] ,flux[8] ,flux[9] ,    # intensity/flux related 
            flux[10],flux[11],flux[12],flux[13],flux[14],flux[15],flux[16],flux[17],flux[18],flux[19],
            flux[20],flux[21],flux[22],flux[23],flux[24],flux[25],flux[26],flux[27],flux[28],flux[29], 
            flux[30],flux[31],flux[32],flux[33],flux[34],flux[35],flux[36],flux[37],flux[38],flux[39],
            
            
            f_buffer[0] ,f_buffer[1] ,f_buffer[2] ,f_buffer[3] ,f_buffer[4] ,f_buffer[5] ,f_buffer[6] ,f_buffer[7] ,f_buffer[8] ,f_buffer[9] ,    # intensity/f_buffer related 
            f_buffer[10],f_buffer[11],f_buffer[12],f_buffer[13],f_buffer[14],f_buffer[15],f_buffer[16],f_buffer[17],f_buffer[18],f_buffer[19],
            f_buffer[20],f_buffer[21],f_buffer[22],f_buffer[23],f_buffer[24],f_buffer[25],f_buffer[26],f_buffer[27],f_buffer[28],f_buffer[29], 
            f_buffer[30],f_buffer[31],f_buffer[32],f_buffer[33],f_buffer[34],f_buffer[35],f_buffer[36],f_buffer[37],f_buffer[38],f_buffer[39]
            
        ]
        ,
        
        names =
        (
            'X','Y', 'X_image','Y_image',                                                       # positional
            
            'area_cnt', 'area_minCircle', 'area_ellipse',                                       # size-related
            
            'ell_angle','ell_smaj','ell_smin','ell_e',                                          # fitted ellipse
     
            'flux_01','flux_02','flux_03','flux_04','flux_05','flux_06','flux_07','flux_08','flux_09',              # intensity/flux related 
            'flux_10','flux_11','flux_12','flux_13','flux_14','flux_15','flux_16','flux_17','flux_18','flux_19', 
            'flux_20','flux_21','flux_22','flux_23','flux_24','flux_25','flux_26','flux_27','flux_28','flux_29', 
            'flux_30','flux_31','flux_32','flux_33','flux_34','flux_35','flux_36','flux_37','flux_38','flux_39',
            'flux_40',
            
                          'f_buffer_01','f_buffer_02','f_buffer_03','f_buffer_04','f_buffer_05','f_buffer_06','f_buffer_07','f_buffer_08','f_buffer_09',    # intensity/f_buffer related 
            'f_buffer_10','f_buffer_11','f_buffer_12','f_buffer_13','f_buffer_14','f_buffer_15','f_buffer_16','f_buffer_17','f_buffer_18','f_buffer_19', 
            'f_buffer_20','f_buffer_21','f_buffer_22','f_buffer_23','f_buffer_24','f_buffer_25','f_buffer_26','f_buffer_27','f_buffer_28','f_buffer_29', 
            'f_buffer_30','f_buffer_31','f_buffer_32','f_buffer_33','f_buffer_34','f_buffer_35','f_buffer_36','f_buffer_37','f_buffer_38','f_buffer_39', 
            'f_buffer_40'            
        )
        )
    
    return t
    

def extract_features_and_update_catalog(img_16bit, cnt, t_final, n_cell, minCntLength, imgW, imgH, all_frames):
    
    # constants
    minPixVal , maxPixVal = 0, 255
    flux_keyword = 'flux'
    flux_feature_columns = [f for f in t_final.colnames if flux_keyword in f]
    
    n_buff = 2 # buffer pixel width within which the flux will be measured around the main cell. If n_buff = 0, it is as usual (no boundary around cell)
    buff_keyword = 'buffer'
    buff_feature_columns = [f for f in t_final.colnames if buff_keyword in f]
    
    # objects should consist of at least 5 connecting points
    if len(cnt) >= minCntLength:
        
        # create a contour object
        c = imcutil.Contour(cnt)
        
        # ellipse = cv2.fitEllipse(c)
        # (x, y), (MA, ma), angle = ellipse
        # ellipse_area = np.pi * (MA/2.0) * (ma/2.0)
        # area.append(ellipse_area)
        # cv2.ellipse(imgOpencv_8bit_copy, ellipse, random_color(), 1)
        # r = lambda: random.randint(0,255)
        
        # if ma < 75:
        #     # cv2.ellipse(imgOpencv_8bit_copy, ellipse, (0,255,0), 1)        
        #     cv2.ellipse(imgOpencv_8bit_copy, ellipse, random_color(), 1)
        
        # draw a circle enclosing the object
        # ((x, y), r) = cv2.minEnclosingCircle(c)
        # # cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 1)
        # cv2.putText(imgOpencv_8bit_copy, "#{}".format(label), (int(x) - 10, int(y)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
        # contour.append(cnt)
        
        # objects should have non-zero area size
        if c.area > 0:
            
            # position-related
            xc , yc = c.centroid[0] , c.centroid[1]
            
            # bounding box
            xRect , yRect , wRect , hRect = cv2.boundingRect(cnt)
            
            # objects should be inside field of view
            if (xc - wRect - n_buff)> 0 and (yc - hRect - n_buff) > 0 and (xc + wRect + n_buff) < imgW and (yc + hRect + n_buff) < imgH:
                
                # # Draw contours on mask image
                # # ---------------------------
                # cv2.drawContours(masked_image, [cnt], 0, random_color(), -1)
                # cv2.circle(imgOpencv_8bit_copy, (int(xc), int(yc)), 1, (0, 255, 255), -1)
                
                # Positional
                # ----------
                            
                # local
                t_final['X_image'][n_cell] = xc
                t_final['Y_image'][n_cell] = yc #imgH - yc # Useful for TOPCAT otherwise 'yc' only

                t_final['X'][n_cell] = xc
                t_final['Y'][n_cell] = imgH - yc            
           
                # Morphological
                # -------------
                
                # area related
                t_final['area_cnt'][n_cell] = c.area
                t_final['area_minCircle'][n_cell] = c.mincircle_area
                t_final['area_ellipse'][n_cell]  = c.area_ellipse
        
        
                # fitted ellipse
                t_final['ell_angle'][n_cell]= c.rotation_angle
                t_final['ell_smaj'][n_cell]= c.majoraxis_length
                t_final['ell_smin'][n_cell]= c.minoraxis_length
                t_final['ell_e'][n_cell]= c.eccentricity
                
                
                # flux related
                # ------------
                # xRect , yRect , wRect , hRect = cv2.boundingRect(cnt)
                
                # First mask individual cells (segmented object)
                kernel = np.ones((3,3),np.uint8)
                mask_fast = np.zeros( (hRect + 2 * n_buff , wRect + 2 * n_buff), dtype=np.uint8)
                cv2.drawContours(mask_fast, [np.subtract( cnt , ( xRect - n_buff , yRect - n_buff) ) ] , minPixVal , maxPixVal , thickness = -1)
                mask_fast_dilation = cv2.dilate(mask_fast,kernel,iterations = n_buff)
                mask_buff = cv2.subtract(mask_fast_dilation, mask_fast)
                
                # examine sample mask images
                if n_cell == 0:
                    cv2.imwrite('./x_test_mask.tif', mask_fast)
                    cv2.imwrite('./x_test_mask_dialation.tif', mask_fast_dilation)                    
                    cv2.imwrite('./x_test_mask_buffer.tif', mask_buff)         
                    
                
                # crop_src = imgPIL_channels[ch_index][yRect:(yRect + hRect), xRect:(xRect + wRect)]
                # flux[ch_index][n_cell] = cv2.mean(crop_src ,  mask = mask_fast)[0]
                
                # crop and extract the flux from REF frame
                # crop_src = img_16bit[yRect:(yRect + hRect), xRect:(xRect + wRect)]
                # t_final['flux_01'][n_cell]= cv2.mean(crop_src ,  mask = mask_fast)[0]
                
                # extract flux from all available channels
                for ch_index in range( len(all_frames) ):
                    crop_src = all_frames[ch_index][yRect - n_buff :(yRect + hRect + n_buff ), xRect - n_buff :(xRect + wRect + n_buff)]
                    t_final[ flux_feature_columns[ch_index] ][n_cell] = cv2.mean(crop_src ,  mask = mask_fast)[0]
                    t_final[ buff_feature_columns[ch_index] ][n_cell] = cv2.mean(crop_src ,  mask = mask_buff)[0]
                n_cell += 1
                
    return n_cell


def random_color():
    r = lambda: random.randint(0,255)
    # rgbl=[255,0,0]
    # random.shuffle(rgbl)
    return (r(),r(),r())








def get_binary_image(img_8bit):
    

    maxPixVal = 2**8 - 1

    # denoise image
    gb_ksize = 0    # Gaussian Blur parameters
    gb_sigma = 2.0  # Gaussian Blur parameters
            
    img_8bitDenoised = cv2.GaussianBlur(img_8bit,
                                        (gb_ksize , gb_ksize) ,
                                        sigmaX = gb_sigma ,
                                        sigmaY = gb_sigma
                                        )
    # binarize & remove background
    adapThresh_blcokSize= 15
    adapThresh_constant = -7.5
     
    img_binary = cv2.adaptiveThreshold(img_8bitDenoised, maxPixVal,
                                              cv2.ADAPTIVE_THRESH_MEAN_C,
                                              cv2.THRESH_BINARY,
                                              adapThresh_blcokSize,
                                              adapThresh_constant)

    return img_binary





def apply_wShed_and_get_cluster_labels(img_8bit, img_binary):

    '''
    # ----------------------------------------------------------------------------------------------------               
    #                            W A T E R S H E D      S E G M E N T A T I O N  ( START )
    # ----------------------------------------------------------------------------------------------------
    
    
    
    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    '''

    img_binary = get_binary_image(img_8bit)
    
    D = ndimage.distance_transform_edt(img_binary)
    
    
    localMax = peak_local_max(D, indices=False, min_distance=3,labels=img_binary)
    
     
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=img_binary) # If returned warning, upgrade to latest Skimage


    return labels







# preparing the final output catalog
def create_t_final(labels, img_16bit, all_frames):
    

    '''
    # loop over the unique labels returned by the Watershed

    # labels: It is the original image where individual objects detected through watershed segmentation are assigned unique ID.
    # For example, all x,y indices having ID = 35 belongs to one cell. Therefore having estimated the 'labels' array, now we have to
    # go through all of those unique IDs and extract x,y indices associated to those IDs and then create small contour images in order
    # to pass it into cv2.findContour and then extract other information for each objects such as centroid, area, ellipticity etc.
    '''



    minCntLength = 5
    imgH, imgW = img_16bit.shape
    n_cell = 0 # initialize cell counts
    
    # construct a spare matrix: It is as if one does a flattenning of the 'labels' array and therefore it is much faster (x50 times)
    sp_arr = csr_matrix(labels.reshape(1,-1))
    labels_shape = labels.shape
    
    cluster_index_total = len(np.unique(sp_arr.data))
    cluster_index_list = np.arange(1, cluster_index_total + 1 ) # excluding index = [background]

    # create an emoty table (output catalog)
    t_final = get_feature_table(cluster_index_total)
    
    # go through non-zero values in the sparsed version of the 'labels' array
    # for i in np.unique(sp_arr.data)[1:10]:  # as a bonus this `unique` call should be faster too
    for cluster_index in cluster_index_list: # OR for i in np.unique(sp_arr.data)
            
        # get mask image and its top-left corner position
        cnt_mask , cnt_topLeft_P = get_cnt_mask(cluster_index, sp_arr, labels_shape)
        
        # detect contours in the cnt_mask and grab the largest one
        cnt = get_contour_in_mask(cnt_mask, cnt_topLeft_P)
        
        # extract features from individual contours (cnt) and add into t_final - Also update n_cell counter
        n_cell = extract_features_and_update_catalog(img_16bit, cnt, t_final, n_cell, minCntLength, imgW, imgH, all_frames)
    
        # Progress bar
        # ------------
        # count += 1
        progress = 100.0*(float(n_cell)/len(cluster_index_list))
        sys.stdout.write("\tProgress (w-shed): %d%%  \r" % (progress) )
        sys.stdout.flush()
        
   
    return t_final

def create_16bit_mask_image(labels):
    
    # convert from int32 to uint16
    mask = labels.astype(np.uint16)
    
    # construct an image
    im = Image.fromarray(mask)
    
    return im


# convert PIL frame to opencv (16-bit)
def convert_PIL_16bit_to_opencv_16bit(frame):
    return np.array(frame)

# convert opencv_16bit to normalized opencv 8bit
def convert_opencv_16bit_to_normalizedOpencv_8bit(imgOpencv_16bit, normalized_factor):
    
    imgOpencv_16bitNormal = cv2.normalize(imgOpencv_16bit, dst=None, alpha=0, beta=2**16, norm_type=cv2.NORM_MINMAX)
    imgOpencv_8bit = bytescale(imgOpencv_16bitNormal * normalized_factor)
    
    return imgOpencv_8bit

# create a draft image for visualisation only
def create_draft_RGB_image_for_visualization(imgOpencv_8bit):
    
    imgOpencv_8bit_copy = imgOpencv_8bit.copy()
    imgOpencv_8bit_copy = cv2.cvtColor(imgOpencv_8bit_copy, cv2.COLOR_GRAY2BGR)
    imgOpencv_8bit_copy = cv2.applyColorMap(imgOpencv_8bit_copy, cv2.COLORMAP_PINK)
    imgOpencv_8bit_copy = cv2.GaussianBlur(imgOpencv_8bit_copy,(5,5),0)
    imgOpencv_8bit_copy = cv2.GaussianBlur(imgOpencv_8bit_copy,(0,0),sigmaX = 0.5, sigmaY = 0.5)
    
    return imgOpencv_8bit_copy

class ImageSequence:
    '''
    Going through individual image (each time it is called) in an image sequence.
        
    Further reading:
    http://pillow.readthedocs.org/en/3.1.x/handbook/tutorial.html#reading-sequences
    '''
    def __init__(self, im):
        self.im = im
    def __getitem__(self, ix):
        try:
            if ix:
                self.im.seek(ix)
            return self.im
        except EOFError:
            raise IndexError # end of sequence


def get_ref_channel_opencv_8bit_normalized(ref_frame, normalized_factor):
    
    # img_channel, ref_frame = get_tot_channel_number_and_specific_slice(imgPIL_cube, ref_channel)
            
    imgOpencv_16bit = convert_PIL_16bit_to_opencv_16bit(ref_frame)
    
    imgOpencv_8bit  = convert_opencv_16bit_to_normalizedOpencv_8bit(imgOpencv_16bit, normalized_factor)                                                                        
    # imgOpencv_8bit_copy  = create_draft_RGB_image_for_visualization(imgOpencv_8bit)
    
    return imgOpencv_16bit, imgOpencv_8bit #, imgOpencv_8bit_copy

# find number of channels
def get_tot_channel_number_and_specific_slice(imgPIL_cube, ref_channel):

    # list containing all slices
    all_frames = list()
    
    for index, frame in enumerate(ImageSequence(imgPIL_cube)):
    
        img_channel = index + 1
        
        all_frames.append( np.array(frame) )
        #                  <------------>
        #                  PIL to numpy (opencv form)
        
        if img_channel == ref_channel: ref_frame = frame
    
    return img_channel, ref_frame, all_frames
        
        
def get_fName(str):
    '''Return the name of a file.'''
    fileName , fileExtension = os.path.splitext(str)
    fileName = ntpath.basename(fileName)
    return fileName
    
    
def get_pseudo_opecv_8bit_flat_image(imgOpencv_16bit, normalized_factor):
    
    # float means the array only e.g. float16 - So it is not an image (which is uint8, uint16, or uint32)
   imgOpencv_16bit_float = imgOpencv_16bit.astype(np.float16) 

   imgOpencv_16bit_filtered =  gaussian_filter(imgOpencv_16bit, sigma=5)
   imgOpencv_16bit_filtered_float = imgOpencv_16bit_filtered.astype(np.float16)        
   imgOpencv_16bit_filtered_float_normalized = imgOpencv_16bit_filtered_float / np.mean(imgOpencv_16bit_filtered)
   imgOpencv_16bit_normalized = imgOpencv_16bit_filtered_float_normalized.astype(np.uint16)
   
   eps_factor = 1000.0 # get rid of zeros in array
   imgOpencv_16bit_float[ np.where(imgOpencv_16bit_float == 0.0) ] = 1.0 / eps_factor
   imgOpencv_16bit_filtered_float_normalized[ np.where(imgOpencv_16bit_filtered_float_normalized == 0.0) ] = eps_factor
   
   imgOpencv_16bit_float_flat = imgOpencv_16bit_float / imgOpencv_16bit_filtered_float_normalized 
   imgOpencv_16bit_flat = imgOpencv_16bit_float_flat.astype(np.uint16)
   imgOpencv_8bit_flat_normalized = convert_opencv_16bit_to_normalizedOpencv_8bit(imgOpencv_16bit_flat, normalized_factor)
   
   return imgOpencv_8bit_flat_normalized
    


def process_image(img_file, ref_channel, normalized_factor, outputPath_ref, outputPath_mask, outputPath_cat,imgFormat ,imgFormatOut, catFormat):
# time.sleep(1)
        
        # read 16-bit data cube
        imgPIL_cube = Image.open(img_file)
        img_name = get_fName(img_file)
    
            
        # extract:
        # (i) total number of channels in the current PIL cube data <-- to be used in 'get_feature_table' function
        # (ii) reference frame to be usd for nuclear segmentation (visual inspection of channels) <-- for nuclear segmentation
        # (iii) a lis where each element is numpy array version of each slice in the image data cube <-- for for flux extraction    
        n_channel_tot , ref_frame, all_frames = get_tot_channel_number_and_specific_slice(imgPIL_cube, ref_channel)
        print ('\n',img_file, '\tn_tot_channel: ', n_channel_tot,'\n')
        
        # imgOpencv_16bit & imgOpencv_8bit are 16-bit and 8-bit opencv versions of the ref_frame
        imgOpencv_16bit, imgOpencv_8bit = get_ref_channel_opencv_8bit_normalized(ref_frame, normalized_factor)
        
        # # for visualization in topcat:
        # imgOpencv_8bit_copy = create_draft_RGB_image_for_visualization(imgOpencv_8bit)
 
        # create pseudo_flat_field_corrected ocv_8-bit image from ocv_16-bit image        
        imgOpencv_8bit_flat_normalized = get_pseudo_opecv_8bit_flat_image(imgOpencv_16bit, normalized_factor)
        cv2.imwrite(outputPath_ref + 'flat_' + img_name + imgFormat, imgOpencv_8bit_flat_normalized)        
        
        # visualization purpose
        cv2.imwrite(outputPath_ref + img_name + imgFormatOut,imgOpencv_8bit)
        
        # cv2.imwrite('./test_ocv_draft.jpg',imgOpencv_8bit_copy)    
        # img_16bit = cv2.imread(img_file, -1)
        # 
        # # convert to 8bit
        # # img_8bit = imutil.map_uint16_to_uint8_skimage(img_16bit)
        # img_8bit = imutil.map_uint16_to_uint8_scipy(img_16bit)
        # 
        # 
        img_binary = get_binary_image(imgOpencv_8bit_flat_normalized) # <---------------------------------------------- change here
        
        labels = apply_wShed_and_get_cluster_labels(imgOpencv_8bit_flat_normalized, img_binary) # <-------------------- change here
        
        mask_img = create_16bit_mask_image(labels)
        mask_img.save(outputPath_mask + img_name + '_mask.'  + imgFormat)
        
        t_final = create_t_final(labels, imgOpencv_16bit, all_frames)
        
        print ('\tWriting output table and masked image...\n\n')
        
        # Refining the output table catalog: removing unused rows
        t_final.remove_rows( np.where(t_final['X']==0)[0] )
        t_final.write(outputPath_cat + img_name + catFormat,  overwrite=True)
        # cv2.imwrite('./test_mask.jpg', masked_image)
        return outputPath_cat + img_name + catFormat
    
        
'''
########################################### I M C   F U N C T I O N S    E N D  ##############################################
'''





'''
-----------------
F U N C T I O N S
-----------------
'''


def map_uint16_to_uint8_skimage(img_16bit):

    '''
    Converting the input 16-bit image to uint8 dtype (using Scikit-Image)
    
    reference:
    http://scikit-image.org/docs/dev/user_guide/data_types.html
     
    '''

    # library
    from skimage import img_as_ubyte
    import warnings
        
    # conversion
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img_8bit = img_as_ubyte(img_16bit)
        
    return img_8bit


def map_uint16_to_uint8_scipy(img_16bit):

    '''
    Converting the input 16-bit image to uint8 dtype (using Scipy)
    
    reference:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.bytescale.html#scipy.misc.bytescale
    
    see also the source code:
    https://github.com/scipy/scipy/blob/master/scipy/misc/pilutil.py
    '''
    
    # library
    from scipy.misc import bytescale
    
    # conversion
    img_8bit = bytescale(img_16bit)
    
    return img_8bit







'''
-------------
C L A S S E S
-------------
'''

class Grid:
    """
    By Ali:
    
    A class to generate positional information in order 
    to split a large image into smaller tiles/chunks.
    

    Desing

    (A) = chunksOverlap_pt1 :(x_overlapTopLeft                  , y_overlapTopLeft                   )
    (B) = chunks_pt1        :(x_topLeft                         , y_topLeft                          )
    (C) = chunks_pt2        :(x_topLeft        + ch_width       , y_topLeft        + ch_height       )
    (D) = chunksOverlap_pt2 :(x_overlapTopLeft + ch_overlapWidth, y_overlapTopLeft + ch_overlapHeight)


                  <================= ch_overlapWidth =================>

             d_overlapLeft_w            ch_width             d_overlapRight_w
                  <====> <=====================================> <====>                   
                      
                 (A)
                  .....................................................
                  .     |                                       |     .
                  .     |                                       |     .
                  .......---------------------------------------.......
                  .     |(B)                                    |     .
                  .     |                                       |     .
                  .     |                                       |     .
                  .     |                                       |     .
                  .     |                                       |     .
                  .     |                                       |     .
                  .     |                                       |     .
                  .     |                                       |     .
                  .     |                                       |     .
                  .     |                                       |     .
                  .     |                                       |     .
                  .     |                                       |     .
                  .     |                                       |     .
                  .     |                                       |     .
                  .     |                                       |     .
                  .     |                                   (C) |     .
                  ......---------------------------------------........
                  .     |                                       |     .
                  .     |                                       |     .
                  .....................................................
                                                                      (D)  

    
    """
    
    def __init__(self, img, tile_h = 1000, tile_w = 1000, tileOverlap_h = 25, tileOverlap_w = 25):
        
        ''' Default tile dimension is 1000 x 1000, i.e. 1.0 Megapixel 
        with 25 pixels overlap along each side'''
        
        # input matter
        self.img = img
        self.chunk_w = tile_w
        self.chunk_h = tile_h
        self.chunkOverlap_w = tileOverlap_w
        self.chunkOverlap_h = tileOverlap_h
        self.img_h = img.shape[0]
        self.img_w = img.shape[1]
        self.n_chunk_h = int(self.img_h/self.chunk_h) + 1
        self.n_chunk_w = int(self.img_w/self.chunk_w) + 1
        self.n_chunk_total_complete = (self.n_chunk_h - 1) * (self.n_chunk_w - 1)
        self.n_chunk_total = self.n_chunk_h * self.n_chunk_w
    
    
    def _grid_parameters(self):
        
        chunks_id = []
        chunks_topLeft_x = []
        chunks_topLeft_y = []
        chunks_height = []
        chunks_width  = []
        chunks_pt1 = []
        chunks_pt2 = []

        chunksOverlap_topLeft_x = []
        chunksOverlap_topLeft_y = []
        chunksOverlap_height = []
        chunksOverlap_width  = []
        chunksOverlap_pt1 = []
        chunksOverlap_pt2 = []
        
        n_chunk = 0 # initialise a chunk ID

        
        # iterate (up -> down)
        for j in range(self.n_chunk_h):

            # iterate (left -> right)
            for i in range(self.n_chunk_w):

                n_chunk += 1

                #  --------------
                # | Chunk itself |
                #  --------------
                # ===================================================================

                # adjust for the last chunk width along 'w' direction (left -> right)
                # -------------------------------------------------------------------

                # check if we reach to the last chunk on the right side of the image
                if (i * self.chunk_w + self.chunk_w) > self.img_w:
                    final_chunk_w = self.img_w - ( (i + 1) * self.chunk_w) - 1
                else:
                    final_chunk_w = 0


                # adjust for the last chunk height along 'h' direction (up -> down)
                # -----------------------------------------------------------------

                # check if we reach to the last chunk on the bottom of the image
                if (j * self.chunk_h + self.chunk_h) > self.img_h:
                    final_chunk_h = self.img_h - ( (j + 1) * self.chunk_h) - 1
                else:
                    final_chunk_h = 0


                # output info
                chunks_id.append(n_chunk)

                x_topLeft , y_topLeft = i * self.chunk_w , j * self.chunk_h
                ch_width, ch_height = (self.chunk_w + final_chunk_w) , (self.chunk_h + final_chunk_h)

                chunks_topLeft_x.append(x_topLeft)
                chunks_topLeft_y.append(y_topLeft)

                chunks_width.append(ch_width)
                chunks_height.append(ch_height)

                chunks_pt1.append((x_topLeft , y_topLeft))
                chunks_pt2.append( ( x_topLeft + ch_width,  y_topLeft + ch_height) )


                #  ---------------
                # | Chunk overlap |
                #  ---------------
                # ===================================================================
                
                # stage.1: Determining d_overlap Left, Right, Top, and Bottom

                # extereme left overlap
                if (x_topLeft - self.chunkOverlap_w) <= 0 :
                    d_overlapLeft_w = 0
                else:
                    d_overlapLeft_w = self.chunkOverlap_w


                # extereme right overlap
                if (x_topLeft + ch_width + self.chunkOverlap_w) >= self.img_w:
                    d_overlapRight_w = 0
                else:
                    d_overlapRight_w = self.chunkOverlap_w


                # extereme top
                if (y_topLeft - self.chunkOverlap_h) <= 0 :
                    d_overlapTop_h = 0
                else:
                    d_overlapTop_h = self.chunkOverlap_h


                # extereme bottom
                if (y_topLeft + ch_height + self.chunkOverlap_h) >= self.img_h:
                    d_overlapBottom_h = 0
                else:
                    d_overlapBottom_h = self.chunkOverlap_h

                # stage.2: chunksOverlap information

                x_overlapTopLeft = x_topLeft - d_overlapLeft_w 
                y_overlapTopLeft = y_topLeft - d_overlapTop_h
                ch_overlapWidth  = d_overlapLeft_w + ch_width  + d_overlapRight_w
                ch_overlapHeight = d_overlapTop_h  + ch_height + d_overlapBottom_h

                chunksOverlap_topLeft_x.append(x_overlapTopLeft)
                chunksOverlap_topLeft_y.append(y_overlapTopLeft)
                chunksOverlap_height.append(ch_overlapWidth)
                chunksOverlap_width.append(ch_overlapHeight)
                chunksOverlap_pt1.append( (x_overlapTopLeft, y_overlapTopLeft) )
                chunksOverlap_pt2.append( (x_overlapTopLeft + ch_overlapWidth, y_overlapTopLeft + ch_overlapHeight) )
        
        return  (chunks_id, chunks_topLeft_x, chunks_topLeft_y, chunks_height, chunks_width, chunks_pt1, chunks_pt2, chunksOverlap_topLeft_x, chunksOverlap_topLeft_y, chunksOverlap_height, chunksOverlap_width, chunksOverlap_pt1, chunksOverlap_pt2)             
    
    def get_chunks_id(self):
        return self._grid_parameters()[0]
    
    def get_chunks_topLeft_x(self):
        return self._grid_parameters()[1]
    
    def get_chunks_topLeft_y(self):
        return self._grid_parameters()[2]
    
    def get_chunks_height(self):
        return self._grid_parameters()[3]
    
    def get_chunks_width(self):
        return self._grid_parameters()[4]
    
    def get_chunks_pt1(self):
        return self._grid_parameters()[5]
    
    def get_chunks_pt2(self):
        return self._grid_parameters()[6]
    
    def get_chunksOverlap_topLeft_x(self):
        return self._grid_parameters()[7]
    
    def get_chunksOverlap_topLeft_y(self):
        return self._grid_parameters()[8]
    
    def get_chunksOverlap_height(self):
        return self._grid_parameters()[9]
    
    def get_chunksOverlap_width(self):
        return self._grid_parameters()[10]
    
    def get_chunksOverlap_pt1(self):
        return self._grid_parameters()[11]

    def get_chunksOverlap_pt2(self):
        return self._grid_parameters()[12]

    

class Contour:
    ''' Provides detailed parameter informations about a contour

        Create a Contour instant as follows: c = Contour(contour)
                where src_img should be grayscale image.

        Attributes:

        c.area -- gives the area of the region
        c.parameter -- gives the perimeter of the region
        c.moments -- gives all values of moments as a dict
        c.centroid -- gives the centroid of the region as a tuple (x,y)
        c.bounding_box -- gives the bounding box parameters as a tuple => (x,y,width,height)
        c.bx,c.by,c.bw,c.bh -- corresponds to (x,y,width,height) of the bounding box
        c.aspect_ratio -- aspect ratio is the ratio of width to height
        c.equi_diameter -- equivalent diameter of the circle with same as area as that of region
        c.extent -- extent = contour area/bounding box area
        c.convex_hull -- gives the convex hull of the region
        c.convex_area -- gives the area of the convex hull
        c.solidity -- solidity = contour area / convex hull area
        c.center -- gives the center of the ellipse
        c.majoraxis_length -- gives the length of major axis
        c.minoraxis_length -- gives the length of minor axis
        c.orientation -- gives the orientation of ellipse
        c.eccentricity -- gives the eccentricity of ellipse
        '''
    def __init__(self,cnt):
        self.cnt = cnt
        self.size = len(cnt)

        # MAIN PARAMETERS

        #Contour.area - Area bounded by the contour region'''
        self.area = cv2.contourArea(self.cnt)

        # contour perimeter
        self.perimeter = cv2.arcLength(cnt,True)

        # centroid
        self.moments = cv2.moments(cnt)
        if self.moments['m00'] != 0.0:
            self.cx = self.moments['m10']/self.moments['m00']
            self.cy = self.moments['m01']/self.moments['m00']
            self.centroid = (self.cx,self.cy)
        else:
            self.centroid = "Region has zero area"

        # bounding box
        self.bounding_box=cv2.boundingRect(cnt)
        (self.bx,self.by,self.bw,self.bh) = self.bounding_box

        # aspect ratio
        self.aspect_ratio = self.bw/float(self.bh)

        # equivalent diameter
        self.equi_diameter = np.sqrt(4*self.area/np.pi)

        # extent = contour area/boundingrect area
        self.extent = self.area/(self.bw*self.bh)

        # Minimum Enclosing Circle
        (self.x_mincircle,self.y_mincircle),self.radius_mincircle = cv2.minEnclosingCircle(cnt)
        self.center_mincircle = (int(self.x_mincircle),int(self.y_mincircle))
        self.mincircle_area = np.pi*self.radius_mincircle*self.radius_mincircle

        # Checking Convexity
        self.convexity = cv2.isContourConvex(cnt)

        ### CONVEX HULL ###

        # convex hull
        self.convex_hull = cv2.convexHull(cnt)

        # convex hull area
        self.convex_area = cv2.contourArea(self.convex_hull)

        # solidity = contour area / convex hull area
        if self.convex_area != 0:
            self.solidity = self.area/float(self.convex_area)
        else: self.solidity = 0


        ### ELLIPSE  ###

        self.ellipse = cv2.fitEllipse(cnt)

        # center, axis_length and orientation of ellipse
        (self.center,self.axes,self.orientation) = self.ellipse

        # length of MAJOR and minor axis
        self.majoraxis_length = max(self.axes)
        self.minoraxis_length = min(self.axes)
        self.area_ellipse = np.pi*(self.majoraxis_length/2.)*(self.minoraxis_length/2.)
        
        # rotation angle
        self.rotation_angle = self.orientation

        # eccentricity = sqrt( 1 - (ma/MA)^2) --- ma= minor axis --- MA= major axis
        self.eccentricity = np.sqrt(1-(self.minoraxis_length/self.majoraxis_length)**2)

