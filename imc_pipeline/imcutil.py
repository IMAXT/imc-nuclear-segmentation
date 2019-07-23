import logging
import random
from typing import List

import cv2
import numpy as np
from astropy.table import Table
from PIL import Image
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.sparse import csr_matrix
from skimage.feature import peak_local_max
from skimage.morphology import watershed

from imaxt_image.image import TiffImage

from .contour import Contour

log = logging.getLogger('owl.daemon.pipeline')


def get_cnt_mask(cluster_index, sp_arr, labels_shape):
    """[summary]

    Parameters
    ----------
    cluster_index : [type]
        [description]
    sp_arr : [type]
        [description]
    labels_shape : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    cnt_y_index, cnt_x_index = np.unravel_index(
        sp_arr.indices[sp_arr.data == cluster_index], labels_shape
    )

    cnt_x_min, cnt_x_max, cnt_y_min, cnt_y_max = (
        np.min(cnt_x_index),
        np.max(cnt_x_index),
        np.min(cnt_y_index),
        np.max(cnt_y_index),
    )
    cnt_topLeft_P = (cnt_x_min, cnt_y_min)
    cnt_img_h, cnt_img_w = cnt_y_max - cnt_y_min + 1, cnt_x_max - cnt_x_min + 1
    cnt_img_shape = (cnt_img_h, cnt_img_w)
    cnt_mask_x_index = cnt_x_index - cnt_x_min
    cnt_mask_y_index = cnt_y_index - cnt_y_min
    cnt_mask_xy_index = (cnt_mask_y_index, cnt_mask_x_index)
    cnt_mask = np.zeros(cnt_img_shape, dtype='uint8')
    cnt_mask[cnt_mask_xy_index] = 1

    return cnt_mask, cnt_topLeft_P


def get_contour_in_mask(cnt_mask, cnt_topLeft_P):
    """Detect contours in the cnt_mask and grab the largest one

    Parameters
    ----------
    cnt_mask : [type]
        [description]
    cnt_topLeft_P : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # TODO: why do we need to use .copy() ?
    cnts = cv2.findContours(cnt_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts[-2], key=cv2.contourArea)
    c = c.reshape(c.shape[0], c.shape[2])

    # apply offset to account for the correct location of the cell in the image
    c[:, 0] += cnt_topLeft_P[0]
    c[:, 1] += cnt_topLeft_P[1]

    return c


def get_feature_table(n_valid_cnt=0):
    """[summary]

    Parameters
    ----------
    n_valid_cnt : int, optional
        [description] (the default is 0, which [default_description])

    Returns
    -------
    [type]
        [description]
    """
    # initialize number of elements

    # positional parameters
    feature_X_topcat, feature_Y_topcat, feature_X_image, feature_Y_image = [
        np.zeros(shape=n_valid_cnt, dtype=np.uint32) for _ in range(4)
    ]

    # morphological / geometrical parameters
    feature_area_cnt, feature_area_mincircle, feature_area_ellipse = [
        np.zeros(shape=n_valid_cnt, dtype=np.uint32) for _ in range(3)
    ]

    # fitted ellipse
    feature_ell_angle, feature_ell_smaj, feature_ell_smin, feature_ell_e = [
        np.zeros(shape=n_valid_cnt, dtype=np.float32) for _ in range(4)
    ]

    # intensity parameters: There are 'ch' number of channels for each image
    ch = (
        40
    )  # for the time being, set it to 40. Alternatively, we can read it from the number of available channels in the input data cube
    flux = np.zeros((ch, n_valid_cnt), dtype=np.float32)
    f_buffer = np.zeros((ch, n_valid_cnt), dtype=np.float32)

    # creating binary table
    t = Table(
        [
            feature_X_topcat,
            feature_Y_topcat,
            feature_X_image,
            feature_X_image,  # positional-related
            feature_area_cnt,
            feature_area_mincircle,
            feature_area_ellipse,  # size-related
            feature_ell_angle,
            feature_ell_smaj,
            feature_ell_smin,
            feature_ell_e,  # fitted ellipse
            flux[0],
            flux[1],
            flux[2],
            flux[3],
            flux[4],
            flux[5],
            flux[6],
            flux[7],
            flux[8],
            flux[9],  # intensity/flux related
            flux[10],
            flux[11],
            flux[12],
            flux[13],
            flux[14],
            flux[15],
            flux[16],
            flux[17],
            flux[18],
            flux[19],
            flux[20],
            flux[21],
            flux[22],
            flux[23],
            flux[24],
            flux[25],
            flux[26],
            flux[27],
            flux[28],
            flux[29],
            flux[30],
            flux[31],
            flux[32],
            flux[33],
            flux[34],
            flux[35],
            flux[36],
            flux[37],
            flux[38],
            flux[39],
            f_buffer[0],
            f_buffer[1],
            f_buffer[2],
            f_buffer[3],
            f_buffer[4],
            f_buffer[5],
            f_buffer[6],
            f_buffer[7],
            f_buffer[8],
            f_buffer[9],  # intensity/f_buffer related
            f_buffer[10],
            f_buffer[11],
            f_buffer[12],
            f_buffer[13],
            f_buffer[14],
            f_buffer[15],
            f_buffer[16],
            f_buffer[17],
            f_buffer[18],
            f_buffer[19],
            f_buffer[20],
            f_buffer[21],
            f_buffer[22],
            f_buffer[23],
            f_buffer[24],
            f_buffer[25],
            f_buffer[26],
            f_buffer[27],
            f_buffer[28],
            f_buffer[29],
            f_buffer[30],
            f_buffer[31],
            f_buffer[32],
            f_buffer[33],
            f_buffer[34],
            f_buffer[35],
            f_buffer[36],
            f_buffer[37],
            f_buffer[38],
            f_buffer[39],
        ],
        names=(
            'X',
            'Y',
            'X_image',
            'Y_image',  # positional
            'area_cnt',
            'area_minCircle',
            'area_ellipse',  # size-related
            'ell_angle',
            'ell_smaj',
            'ell_smin',
            'ell_e',  # fitted ellipse
            'flux_01',
            'flux_02',
            'flux_03',
            'flux_04',
            'flux_05',
            'flux_06',
            'flux_07',
            'flux_08',
            'flux_09',  # intensity/flux related
            'flux_10',
            'flux_11',
            'flux_12',
            'flux_13',
            'flux_14',
            'flux_15',
            'flux_16',
            'flux_17',
            'flux_18',
            'flux_19',
            'flux_20',
            'flux_21',
            'flux_22',
            'flux_23',
            'flux_24',
            'flux_25',
            'flux_26',
            'flux_27',
            'flux_28',
            'flux_29',
            'flux_30',
            'flux_31',
            'flux_32',
            'flux_33',
            'flux_34',
            'flux_35',
            'flux_36',
            'flux_37',
            'flux_38',
            'flux_39',
            'flux_40',
            'f_buffer_01',
            'f_buffer_02',
            'f_buffer_03',
            'f_buffer_04',
            'f_buffer_05',
            'f_buffer_06',
            'f_buffer_07',
            'f_buffer_08',
            'f_buffer_09',  # intensity/f_buffer related
            'f_buffer_10',
            'f_buffer_11',
            'f_buffer_12',
            'f_buffer_13',
            'f_buffer_14',
            'f_buffer_15',
            'f_buffer_16',
            'f_buffer_17',
            'f_buffer_18',
            'f_buffer_19',
            'f_buffer_20',
            'f_buffer_21',
            'f_buffer_22',
            'f_buffer_23',
            'f_buffer_24',
            'f_buffer_25',
            'f_buffer_26',
            'f_buffer_27',
            'f_buffer_28',
            'f_buffer_29',
            'f_buffer_30',
            'f_buffer_31',
            'f_buffer_32',
            'f_buffer_33',
            'f_buffer_34',
            'f_buffer_35',
            'f_buffer_36',
            'f_buffer_37',
            'f_buffer_38',
            'f_buffer_39',
            'f_buffer_40',
        ),
    )

    return t


def extract_features_and_update_catalog(
    img_16bit, cnt, t_final, n_cell, minCntLength, imgW, imgH, all_frames, n_buff
):
    """[summary]

    Parameters
    ----------
    img_16bit : [type]
        [description]
    cnt : [type]
        [description]
    t_final : [type]
        [description]
    n_cell : [type]
        [description]
    minCntLength : [type]
        [description]
    imgW : [type]
        [description]
    imgH : [type]
        [description]
    all_frames : [type]
        [description]
    n_buff : int
        buffer pixel width within which the flux will be measured around the main cell. If n_buff = 0, it is as usual (no boundary around cell)
    Returns
    -------
    [type]
        [description]
    """

    # constants
    minPixVal, maxPixVal = 0, 255
    flux_keyword = 'flux'
    flux_feature_columns = [f for f in t_final.colnames if flux_keyword in f]

    buff_keyword = 'buffer'
    buff_feature_columns = [f for f in t_final.colnames if buff_keyword in f]

    # objects should consist of at least 5 connecting points
    if len(cnt) >= minCntLength:

        # create a contour object
        c = Contour(cnt)

        # objects should have non-zero area size
        if c.area > 0:

            # position-related
            xc, yc = c.centroid[0], c.centroid[1]

            # bounding box
            xRect, yRect, wRect, hRect = cv2.boundingRect(cnt)

            # objects should be inside field of view
            if (
                (xc - wRect - n_buff) > 0
                and (yc - hRect - n_buff) > 0
                and (xc + wRect + n_buff) < imgW
                and (yc + hRect + n_buff) < imgH
            ):
                # Positional
                # ----------

                # local
                t_final['X_image'][n_cell] = xc
                t_final['Y_image'][
                    n_cell
                ] = yc  # imgH - yc # Useful for TOPCAT otherwise 'yc' only

                t_final['X'][n_cell] = xc
                t_final['Y'][n_cell] = imgH - yc

                # Morphological
                # -------------

                # area related
                t_final['area_cnt'][n_cell] = c.area
                t_final['area_minCircle'][n_cell] = c.mincircle_area
                t_final['area_ellipse'][n_cell] = c.area_ellipse

                # fitted ellipse
                t_final['ell_angle'][n_cell] = c.rotation_angle
                t_final['ell_smaj'][n_cell] = c.majoraxis_length
                t_final['ell_smin'][n_cell] = c.minoraxis_length
                t_final['ell_e'][n_cell] = c.eccentricity

                # flux related
                # ------------
                # xRect , yRect , wRect , hRect = cv2.boundingRect(cnt)

                # First mask individual cells (segmented object)
                kernel = np.ones((3, 3), np.uint8)
                mask_fast = np.zeros(
                    (hRect + 2 * n_buff, wRect + 2 * n_buff), dtype=np.uint8
                )
                cv2.drawContours(
                    mask_fast,
                    [np.subtract(cnt, (xRect - n_buff, yRect - n_buff))],
                    minPixVal,
                    maxPixVal,
                    thickness=-1,
                )
                mask_fast_dilation = cv2.dilate(mask_fast, kernel, iterations=n_buff)
                mask_buff = cv2.subtract(mask_fast_dilation, mask_fast)

                # extract flux from all available channels
                for ch_index in range(len(all_frames)):
                    crop_src = all_frames[ch_index][
                        (yRect - n_buff) : (yRect + hRect + n_buff),
                        (xRect - n_buff) : (xRect + wRect + n_buff),
                    ]
                    t_final[flux_feature_columns[ch_index]][n_cell] = cv2.mean(
                        crop_src, mask=mask_fast
                    )[0]
                    t_final[buff_feature_columns[ch_index]][n_cell] = cv2.mean(
                        crop_src, mask=mask_buff
                    )[0]
                n_cell += 1

    return n_cell


def random_color() -> List[int]:
    """[summary]

    Returns
    -------
    [type]
        [description]
    """
    return [random.randint(0, 255) for i in range(3)]


def get_binary_image(
    img_8bit, gb_ksize, gb_sigma, adapThresh_blockSize, adapThresh_constant
):
    """[summary]

    Parameters
    ----------
    img_8bit : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    maxPixVal = 2 ** 8 - 1

    img_8bitDenoised = cv2.GaussianBlur(
        img_8bit, (gb_ksize, gb_ksize), sigmaX=gb_sigma, sigmaY=gb_sigma
    )

    img_binary = cv2.adaptiveThreshold(
        img_8bitDenoised,
        maxPixVal,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        adapThresh_blockSize,
        adapThresh_constant,
    )

    return img_binary


def apply_wShed_and_get_cluster_labels(
    img_8bit,
    img_binary,
    min_distance,
    gb_ksize,
    gb_sigma,
    adapThresh_blockSize,
    adapThresh_constant,
):
    """Watershed segmentation

    Parameters
    ----------
    img_8bit : [type]
        [description]
    img_binary : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    img_binary = get_binary_image(
        img_8bit, gb_ksize, gb_sigma, adapThresh_blockSize, adapThresh_constant
    )

    D = ndimage.distance_transform_edt(img_binary)

    localMax = peak_local_max(
        D, indices=False, min_distance=min_distance, labels=img_binary
    )

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(
        -D, markers, mask=img_binary
    )  # If returned warning, upgrade to latest Skimage

    return labels


# preparing the final output catalog
def create_t_final(labels, img_16bit, all_frames, n_buff):
    """[summary]

    Parameters
    ----------
    labels : [type]
        [description]
    img_16bit : [type]
        [description]
    all_frames : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    # loop over the unique labels returned by the Watershed

    # labels: It is the original image where individual objects detected through watershed segmentation are assigned unique ID.
    # For example, all x,y indices having ID = 35 belongs to one cell. Therefore having estimated the 'labels' array, now we have to
    # go through all of those unique IDs and extract x,y indices associated to those IDs and then create small contour images in order
    # to pass it into cv2.findContour and then extract other information for each objects such as centroid, area, ellipticity etc.

    minCntLength = 5
    imgH, imgW = img_16bit.shape
    n_cell = 0  # initialize cell counts

    # construct a spare matrix: It is as if one does a flattenning of the 'labels' array and therefore it is much faster (x50 times)
    sp_arr = csr_matrix(labels.reshape(1, -1))
    labels_shape = labels.shape

    cluster_index_total = len(np.unique(sp_arr.data))
    cluster_index_list = np.arange(
        1, cluster_index_total + 1
    )  # excluding index = [background]

    # create an emoty table (output catalog)
    t_final = get_feature_table(cluster_index_total)

    # go through non-zero values in the sparsed version of the 'labels' array
    # for i in np.unique(sp_arr.data)[1:10]:  # as a bonus this `unique` call should be faster too
    for cluster_index in cluster_index_list:  # OR for i in np.unique(sp_arr.data)

        # get mask image and its top-left corner position
        cnt_mask, cnt_topLeft_P = get_cnt_mask(cluster_index, sp_arr, labels_shape)

        # detect contours in the cnt_mask and grab the largest one
        cnt = get_contour_in_mask(cnt_mask, cnt_topLeft_P)

        # extract features from individual contours (cnt) and add into t_final - Also update n_cell counter
        n_cell = extract_features_and_update_catalog(
            img_16bit,
            cnt,
            t_final,
            n_cell,
            minCntLength,
            imgW,
            imgH,
            all_frames,
            n_buff,
        )

    return t_final


def create_mask_image(labels):
    """[summary]

    Parameters
    ----------
    labels : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    # convert from int32 to uint16
    mask = labels.astype(np.uint16)

    # construct an image
    im = Image.fromarray(mask)

    return im


def _normalize_minmax(
    img: np.ndarray, low: int, high: int, dtype=np.uint16
) -> np.ndarray:
    assert low < high
    im_min, im_max = img.min(), img.max()
    scl = (high - low) / (im_max - im_min)
    im_norm = (img - im_min) * scl + low
    return (im_norm.clip(low, high) + 0.5).astype(dtype)


# convert opencv_16bit to normalized opencv 8bit
def normalize_channel(img, norm_factor):
    """[summary]

    Parameters
    ----------
    imgOpencv_16bit : [type]
        [description]
    normalized_factor : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    img_norm = _normalize_minmax(img, 0, 2 ** 16)
    img_8bit = _normalize_minmax(img_norm * norm_factor, 0, 255, dtype=np.uint8)
    return img_8bit


# find number of channels
def get_frames(cube: Image) -> List[np.ndarray]:
    """Extract all channels in an Image data cube

    Parameters
    ----------
    cube
        data cube

    Returns
    -------
    list of images
    """
    # all_frames = [*map(np.array, ImageSequence.Iterator(cube))]
    all_frames = TiffImage(cube).asarray()
    return all_frames


def get_pseudo_opecv_8bit_flat_image(imgOpencv_16bit, normalized_factor):
    """[summary]

    Parameters
    ----------
    imgOpencv_16bit : [type]
        [description]
    normalized_factor : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # float means the array only e.g. float16 - So it is not an image (which is uint8, uint16, or uint32)
    imgOpencv_16bit_float = imgOpencv_16bit.astype(np.float)

    # TODO: explain choice of sigma=5
    imgOpencv_16bit_filtered = gaussian_filter(imgOpencv_16bit, sigma=5)
    imgOpencv_16bit_filtered_float = imgOpencv_16bit_filtered.astype(np.float16)
    imgOpencv_16bit_filtered_float_normalized = (
        imgOpencv_16bit_filtered_float / np.mean(imgOpencv_16bit_filtered)
    )
    # imgOpencv_16bit_normalized = imgOpencv_16bit_filtered_float_normalized.astype(
    #     np.uint16
    # )

    eps_factor = 1000.0  # get rid of zeros in array
    imgOpencv_16bit_float[np.where(imgOpencv_16bit_float == 0.0)] = 1.0 / eps_factor
    imgOpencv_16bit_filtered_float_normalized[
        np.where(imgOpencv_16bit_filtered_float_normalized == 0.0)
    ] = eps_factor

    imgOpencv_16bit_float_flat = (
        imgOpencv_16bit_float / imgOpencv_16bit_filtered_float_normalized
    )
    imgOpencv_16bit_flat = imgOpencv_16bit_float_flat.astype(np.uint16)
    imgOpencv_8bit_flat_normalized = normalize_channel(
        imgOpencv_16bit_flat, normalized_factor
    )

    return imgOpencv_8bit_flat_normalized


def process_image(img_file, n_buff, normalized_factor, segmentation, outputPath):
    """[summary]

    Parameters
    ----------
    img_file : [type]
        [description]
    ref_channel : [type]
        [description]
    normalized_factor : [type]
        [description]
    outputPath : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # read 16-bit data cube
    # img_cube = Image.open(img_file)
    img_name = img_file.name.replace('.tif', '')

    all_frames = get_frames(img_file)

    log.info('Processing %s, n_tot_channel: %s', img_file, len(all_frames))

    ref_channel = segmentation.pop('ref_channel')
    ref_frame = all_frames[ref_channel - 1]
    ref_frame_8bit = normalize_channel(ref_frame, normalized_factor)

    # create pseudo_flat_field_corrected ocv_8-bit image from ocv_16-bit image
    ref_frame_8bit_flat_normalized = get_pseudo_opecv_8bit_flat_image(
        ref_frame, normalized_factor
    )
    out = outputPath / 'reference'
    out.mkdir(exist_ok=True)
    cv2.imwrite(f'{out}/flat_{img_name}.tif', ref_frame_8bit_flat_normalized)
    cv2.imwrite(f'{out}/{img_name}.jpg', ref_frame_8bit)

    img_binary = get_binary_image(
        ref_frame_8bit_flat_normalized,
        segmentation['gb_ksize'],
        segmentation['gb_sigma'],
        segmentation['adapThresh_blockSize'],
        segmentation['adapThresh_constant'],
    )  # <---------------------------------------------- change here

    labels = apply_wShed_and_get_cluster_labels(
        ref_frame_8bit_flat_normalized,
        img_binary,
        segmentation['min_distance'],
        segmentation['gb_ksize'],
        segmentation['gb_sigma'],
        segmentation['adapThresh_blockSize'],
        segmentation['adapThresh_constant'],
    )  # <-------------------- change here

    mask_img = create_mask_image(labels)
    out = outputPath / 'mask'
    out.mkdir(exist_ok=True)
    mask_img.save(f'{out}/{img_name}_mask.tif')

    t_final = create_t_final(labels, ref_frame, all_frames, n_buff)

    log.info('Writing output table and masked image')

    # Refining the output table catalog: removing unused rows
    t_final.remove_rows(np.where(t_final['X'] == 0)[0])
    out = outputPath / 'catalogue'
    out.mkdir(exist_ok=True)
    t_final.write(f'{out}/{img_name}.fits', overwrite=True)
    # cv2.imwrite('./test_mask.jpg', masked_image)
    return f'{out}/{img_name}.fits'


def map_uint16_to_uint8_skimage(img_16bit):
    """[summary]

    Parameters
    ----------
    img_16bit : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    # Converting the input 16-bit image to uint8 dtype (using Scikit-Image)

    # reference:
    # http://scikit-image.org/docs/dev/user_guide/data_types.html

    # # library
    from skimage import img_as_ubyte
    import warnings

    # conversion
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        img_8bit = img_as_ubyte(img_16bit)

    return img_8bit
