import logging
import random
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import xarray as xr
from astropy.table import Table
from dask import delayed
from PIL import Image
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.sparse import csr_matrix
from skimage.feature import peak_local_max
from skimage.morphology import watershed

from .contour import Contour

log = logging.getLogger("owl.daemon.pipeline")


def get_cnt_mask(cluster_index, sp_arr, labels_shape):
    """The function reads cluster index as well as associated data
    in the form of sparse matrix and returns a list of contour masks.

    Parameters
    ----------
    cluster_index : [list]
        list of cluster indices
    sp_arr : [list]
        sparse matrix associated with cluster indices
    labels_shape : [list]
        cluster labels (order) associated with cluster indices

    Returns
    -------
    [list, list]
        Contour masks and their top left positions
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
    cnt_mask = np.zeros(cnt_img_shape, dtype="uint8")
    cnt_mask[cnt_mask_xy_index] = 1

    return cnt_mask, cnt_topLeft_P


def get_contour_in_mask(cnt_mask: List, cnt_topLeft_P: List):
    """Detect contours in the cnt_mask and grab the largest one

    Parameters
    ----------
    cnt_mask
        Contour mask
    cnt_topLeft_P
        Contour top left position

    Returns
    -------
    contour
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
    """Setup an empty output table with feature columns with necessary number
    of rows (=n_valid_cnt).

    The user can change the number of output columns
    using the keyword 'ch' which currently is set to 40.
    If the number of measured feature is less than 'ch', then the rest of
    feature column are set to zero. Else, the value of 'ch' must be increased.
    As a rule of thumb, ch >= (# of calibration channels + # anti-body channels).
    As an example, for an IMC image CUBE with 5 calibration channel and 23 antibody
    channels, ch should be >= 28 (5 + 23).

    Parameters
    ----------
    n_valid_cnt : int, optional
        Number of extracted contours (the default is 0)

    Returns
    -------
    [astropy.table]
        An empty table with number of rows equal to n_valid_cnt and necessary feature columns
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
    ch = 50  # for the time being, set it to 40. Alternatively, we can read it from the number of available channels in the input data cube
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
            flux[40],
            flux[41],
            flux[42],
            flux[43],
            flux[44],
            flux[45],
            flux[46],
            flux[47],
            flux[48],
            flux[49],
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
            f_buffer[40],
            f_buffer[41],
            f_buffer[42],
            f_buffer[43],
            f_buffer[44],
            f_buffer[45],
            f_buffer[46],
            f_buffer[47],
            f_buffer[48],
            f_buffer[49],
        ],
        names=(
            "X",
            "Y",
            "X_image",
            "Y_image",  # positional
            "area_cnt",
            "area_minCircle",
            "area_ellipse",  # size-related
            "ell_angle",
            "ell_smaj",
            "ell_smin",
            "ell_e",  # fitted ellipse
            "flux_01",
            "flux_02",
            "flux_03",
            "flux_04",
            "flux_05",
            "flux_06",
            "flux_07",
            "flux_08",
            "flux_09",  # intensity/flux related
            "flux_10",
            "flux_11",
            "flux_12",
            "flux_13",
            "flux_14",
            "flux_15",
            "flux_16",
            "flux_17",
            "flux_18",
            "flux_19",
            "flux_20",
            "flux_21",
            "flux_22",
            "flux_23",
            "flux_24",
            "flux_25",
            "flux_26",
            "flux_27",
            "flux_28",
            "flux_29",
            "flux_30",
            "flux_31",
            "flux_32",
            "flux_33",
            "flux_34",
            "flux_35",
            "flux_36",
            "flux_37",
            "flux_38",
            "flux_39",
            "flux_40",
            "flux_41",
            "flux_42",
            "flux_43",
            "flux_44",
            "flux_45",
            "flux_46",
            "flux_47",
            "flux_48",
            "flux_49",
            "flux_50",
            "f_buffer_01",
            "f_buffer_02",
            "f_buffer_03",
            "f_buffer_04",
            "f_buffer_05",
            "f_buffer_06",
            "f_buffer_07",
            "f_buffer_08",
            "f_buffer_09",  # intensity/f_buffer related
            "f_buffer_10",
            "f_buffer_11",
            "f_buffer_12",
            "f_buffer_13",
            "f_buffer_14",
            "f_buffer_15",
            "f_buffer_16",
            "f_buffer_17",
            "f_buffer_18",
            "f_buffer_19",
            "f_buffer_20",
            "f_buffer_21",
            "f_buffer_22",
            "f_buffer_23",
            "f_buffer_24",
            "f_buffer_25",
            "f_buffer_26",
            "f_buffer_27",
            "f_buffer_28",
            "f_buffer_29",
            "f_buffer_30",
            "f_buffer_31",
            "f_buffer_32",
            "f_buffer_33",
            "f_buffer_34",
            "f_buffer_35",
            "f_buffer_36",
            "f_buffer_37",
            "f_buffer_38",
            "f_buffer_39",
            "f_buffer_40",
            "f_buffer_41",
            "f_buffer_42",
            "f_buffer_43",
            "f_buffer_44",
            "f_buffer_45",
            "f_buffer_46",
            "f_buffer_47",
            "f_buffer_48",
            "f_buffer_49",
            "f_buffer_50",
        ),
    )

    return t


def extract_features_and_update_catalog(
    img_16bit,
    cnt,
    t_final,
    n_cell,
    minCntLength,
    imgW,
    imgH,
    all_frames,
    n_buff,
    out_mask_each_cell,
    perform_full_analysis,
    ref_frame_8bit_flat_normalized_copy,
):
    """The function, extracts information for each segmented cell and record it in the table already defined. Each piece of information extracted for the same cell, would be assigned to a feature column in the input table. Each row in the table is associated with one cell. Therefore the input table which originally is empty, would be populated with individual cell's information. The function returns as a final output, the number of cells detected.

    Parameters
    ----------
    img_16bit : [numpy array]
        Input IMC image cube (16-bit) for flux or mean pixel intensity estimation.
    cnt : [list]
        List of detected cell contours
    t_final : [astropy table]
        Final output catalog
    n_cell : [int]
        Number of detected cells
    minCntLength : [int]
        Minimum length of a detected contour. This should be always greate than 5.
    imgW : [int]
        IMC image width (constant across all channels for the same slice)
    imgH : [int]
        IMC image height (constant across all channels for the same slice)
    all_frames : [numpy array]
        Input IMC image cube (16-bit) for flux or mean pixel intensity estimation.
    n_buff : int
        buffer pixel width within which the flux will be measured around the main cell. If n_buff = 0, it is as usual (no boundary around cell)
    Returns
    -------
    [int]
        Number of detected cells after filtering out bad ones.
    """

    # constants
    minPixVal, maxPixVal = 0, 255
    flux_keyword = "flux"
    flux_feature_columns = [f for f in t_final.colnames if flux_keyword in f]

    buff_keyword = "buffer"
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
                and hRect > 2
                and wRect > 2
            ):

                # Positional
                # ----------

                # local
                t_final["X_image"][n_cell] = xc
                t_final["Y_image"][n_cell] = yc

                t_final["X"][n_cell] = xc
                t_final["Y"][n_cell] = (
                    imgH - yc
                )  # imgH - yc # Useful for TOPCAT otherwise 'yc' only

                cv2.drawContours(
                    ref_frame_8bit_flat_normalized_copy, [cnt], 0, (0, 255, 0), 1
                )

                # perform this part (extraction of intensities in all channels + shape analysis) if
                # the user request for full analysis by setting the input parameter 'perform_full_analysis'
                # to TRUE in the YAML file. If 'perform_full_analysis' == FALSE, thsi part will be skipped.
                if perform_full_analysis:

                    # Morphological
                    # -------------
                    # area related
                    t_final["area_cnt"][n_cell] = c.area
                    t_final["area_minCircle"][n_cell] = c.mincircle_area
                    t_final["area_ellipse"][n_cell] = int(c.area_ellipse)

                    # fitted ellipse
                    t_final["ell_angle"][n_cell] = c.rotation_angle
                    t_final["ell_smaj"][n_cell] = c.majoraxis_length
                    t_final["ell_smin"][n_cell] = c.minoraxis_length
                    t_final["ell_e"][n_cell] = c.eccentricity

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

                    # output individual image mask
                    cv2.imwrite(
                        f"{out_mask_each_cell}/cell_number_{n_cell}.tif", mask_fast
                    )

                    mask_fast_dilation = cv2.dilate(
                        mask_fast, kernel, iterations=n_buff
                    )
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
    """Creating a list of random, 8-bit, RGB colours.

    Returns
    -------
    [list]
        A list of three RGB values (each range from 0 to 255)
    """
    return [random.randint(0, 255) for i in range(3)]


def get_binary_image(
    img_8bit, gb_ksize, gb_sigma, adapThresh_blockSize, adapThresh_constant
):
    """The functions reads an 8-bit (single channel) image and creates a
    binarised version to be used for segmentation.

    Parameters
    ----------
    img_8bit : [numpy array]
        Single channel numpy array
    gb_ksize
        Gaussian kernel size. ksize.width and ksize.height can differ
        but they both must be positive and odd. Or, they can be zeros
        and then they are computed from sigma* .
    gb_sigma
        Gaussian kernel standard deviation (the same along X and Y).
    adapThresh_blockSize
        Size of a pixel neighborhood that is used to calculate a threshold
        value for the pixel: 3, 5, 7, and so on.
    adapThresh_constant
        Constant subtracted from the mean or weighted mean. Normally, it is
        positive but may be zero or negative as well.

    Returns
    -------
    [numpy array]
        Single channel binarised image (only 0 and 255)
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
    """Apply watershed segmentation

    Parameters
    ----------
    img_8bit : [numpy array]
        Single channel 8-bit image
    img_binary : [numpy array]
        Single channel, 8-bit, binarised (only 0 and 255) image.

    Returns
    -------
    [list]
        Cluster labels
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
def create_t_final(
    labels,
    img_16bit,
    all_frames,
    n_buff,
    out_mask_each_cell,
    perform_full_analysis,
    ref_frame_8bit_flat_normalized_copy,
):
    """Creating the final output catalogue

    Parameters
    ----------
    labels : [list]
        Cluster labels found using watershed segmentation
    img_16bit : [numpy array]
        Input 16-bit, IMC image cube
    all_frames : [type]
        Input 16-bit, IMC image cube

    Returns
    -------
    [astropy table]
        Final output table
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
            out_mask_each_cell,
            perform_full_analysis,
            ref_frame_8bit_flat_normalized_copy,
        )

    return t_final


def create_mask_image(labels):
    """?? TO BE ADDED LATER ??

    Parameters
    ----------
    labels : [list]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    # convert from int32 to uint32
    mask = labels.astype(np.uint32)

    # construct an image
    im = Image.fromarray(mask)

    return im


# clippingng the (100 - 99.7)% of most brightest pixels and
# re-assign their pixel values to the max of 99.7% pixels
def clipping_img16bit_pixel_val(img_16bit, clip_cut_percent=99.7):
    max_val = np.percentile(img_16bit, clip_cut_percent)
    return img_16bit.clip(0, int(max_val))


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
    """This function reads a 16-bit IMC channel and returns a
    normalised 8-bit version of that.

    Parameters
    ----------
    imgOpencv_16bit : [numpy array]
        Single channel IMC, 16-bit image
    ain_image_normalization_factor : [int]
        [10^{-2} percent; Recommended 1 to 50] During the
        processing, the IMC pipeline converts 16-bit images
        into 8-bit and recalculates the pixel values of the
        image so the range is equal to the maximum range for
        the data type. However, to maximise the image contrast,
        some of the pixels are allowed to become saturated.
        Therefore, increasing this value increases the overall contrast.
        If set to 0, there would be no saturated pixels. But in practice,
        this value should be greater than zero to prevent a few outlying
        pixel from causing the histogram stretch to not work as intended.

    Returns
    -------
    [numpy array]
        Single channel IMC, normalised 8-bit channel
    """
    assert img.dtype == "uint16"

    img_norm = _normalize_minmax(img, 0, (2 ** 16 - 1))
    img_8bit = _normalize_minmax(
        img_norm * norm_factor, 0, (2 ** 8 - 1), dtype=np.uint8
    )
    return img_8bit


# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    """
    Automatic brightness and contrast optimization with optional histogram clipping

    Parameters
    ----------
    image: [numpy array]
        Single channel 16-bit or 8-bit image array

    Returns
    -------
    [numpy array]
        Single channel / 8-bit / normalised (if necessary) version of the input image
    """
    assert image.dtype == "uint16"

    # check if image *is not* 8-bit
    if image.max() >= 2 ** 8:

        # clipping outlier pixels
        image = clipping_img16bit_pixel_val(image)

        # convert image to 8-bit
        image = _normalize_minmax(image, 0, (2 ** 8 - 1), dtype=np.uint8)

    # check if input image is single channel
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [2 ** 8], [0, 2 ** 8])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= maximum / 100.0
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = (2 ** 8 - 1) / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    return auto_result


def get_pseudo_opencv_8bit_flat_image(
    imgOpencv_16bit,
    ain_automatic_image_normalization,
    ain_image_normalization_factor,
    aic_apply_intensity_correction,
    aic_sigma,
):
    """The function correct for the observed pixel intensity inhomogeneity across the image

    Parameters
    ----------
    imgOpencv_16bit : [numpy array]
        Single channel 16-bit image
    ain_image_normalization_factor : [int]
        [10^{-2} percent; Recommended 10 to 50] During the processing,
        the IMC pipeline converts 16-bit images into 8-bit and recalculates
        the pixel values of the image so the range is equal to the maximum
        range for the data type. However, to maximise the image contrast,
        some of the pixels are allowed to become saturated. Therefore,
        increasing this value increases the overall contrast. If set to 0,
        there would be no saturated pixels. But in practice, this value should
        be greater than zero to prevent a few outlying pixel from causing the
        histogram stretch to not work as intended.

    Returns
    -------
    [numpy array]
        Intensity corrected, 16-bit single channel, image
    """
    if aic_apply_intensity_correction is False:

        # if automatic normalization key is TRUE
        if ain_automatic_image_normalization is True:

            # Apply auto-normalization
            imgOpencv_8bit_flat_normalized = automatic_brightness_and_contrast(
                imgOpencv_16bit
            )

        # if automatic normalization key is FALSE
        else:

            # Apply manual image normalization based on the normalization factor input parameter
            imgOpencv_8bit_flat_normalized = normalize_channel(
                imgOpencv_16bit, ain_image_normalization_factor
            )
    else:
        # float means the array only e.g. float16 - So it is not an image
        # (which is uint8, uint16, or uint32)
        imgOpencv_16bit_float = imgOpencv_16bit.astype(np.float)
        # TODO: explain choice of sigma=5
        imgOpencv_16bit_filtered = gaussian_filter(imgOpencv_16bit, sigma=aic_sigma)
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
            imgOpencv_16bit_flat, ain_image_normalization_factor
        )
    return imgOpencv_8bit_flat_normalized


@delayed
def sink(
    ref_frame_8bit_flat_normalized,
    ref_frame_8bit_flat_normalized_copy,
    ref_frame_8bit,
    draft_ref_frame_8bit_flat_normalized,
    t_final,
    mask_img,
    name,
    output_path,
):
    output_path.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(f"{output_path}/{name}_ref_flat.tif", ref_frame_8bit_flat_normalized)
    cv2.imwrite(f"{output_path}/{name}_ref.jpg", ref_frame_8bit)

    # Refining the output table catalog: removing unused rows
    log.info("Writing output table")
    t_final.remove_rows(np.where(t_final["X"] == 0)[0])
    t_final.write(f"{output_path}/{name}_cat.fits", overwrite=True)
    t_final.write(f"{output_path}/{name}_cat.csv", overwrite=True)

    log.info("Writing masked image")
    mask_img.save(f"{output_path}/{name}_mask.tif")

    log.info("Writing draft image")
    cv2.imwrite(f"{output_path}/{name}_draft_cnt.tif", ref_frame_8bit_flat_normalized_copy)
    cv2.imwrite(f"{output_path}/{name}_draft.tif", draft_ref_frame_8bit_flat_normalized)

    return True


def process_image(
    input_path: Path, group: str, n_buff: int, segmentation: Dict, output_path: Path
):
    """The function reads segmentation parameters and performs
    watershed segmentation.

    During the process, it does also correct
    for the observed pixel intensity inhomogeneity across the image
    (still under investigation of the source of the existence of such
    an inhomogeneity). The function also record a catalog of detected
    objects (here cell nuclei) on local disk.

    Parameters
    ----------
    img_file : [numpy array]
        Input IMC image
    ref_channel : [int]
        This is the IMC channel to be used as a reference image for
        segmentation (the best nuclear channel)
    ain_image_normalization_factor : [int]
        [10^{-2} percent; Recommended 10 to 50] During the processing,
        the IMC pipeline converts 16-bit images into 8-bit and recalculates
        the pixel values of the image so the range is equal to the maximum
        range for the data type. However, to maximise the image contrast, some
        of the pixels are allowed to become saturated. Therefore, increasing
        this value increases the overall contrast. If set to 0, there would be
        no saturated pixels. But in practice, this value should be greater than
        zero to prevent a few outlying pixel from causing the histogram stretch
        to not work as intended.
    outputPath : [str]
        Location to otput final catalogues of detected cells.

    Returns
    -------
    [str]
        Name of recorded catalog
    """
    # read 16-bit data cube
    # img_name = img_file.name.replace(".tiff", "")

    ds = xr.open_zarr(f"{input_path}", group=group)
    status = ds.attrs["meta"][0]["q_data_source"]
    if status == "invalid":
        return delayed(False)

    ds_q = ds[group].astype("uint16")

    log.info("Processing %s[%s], n_tot_channel: %s", input_path, group, len(ds_q))
    ain_automatic_image_normalization = segmentation[
        "ain_automatic_image_normalization"
    ]
    ain_image_normalization_factor = segmentation["ain_image_normalization_factor"]
    aic_apply_intensity_correction = segmentation["aic_apply_intensity_correction"]
    aic_sigma = segmentation["aic_sigma"]
    ref_channel = segmentation["ref_channel"]
    ref_frame = ds_q[ref_channel - 1].data

    if ain_automatic_image_normalization:
        ref_frame_8bit = delayed(automatic_brightness_and_contrast)(ref_frame)
    else:
        ref_frame_8bit = delayed(normalize_channel)(
            ref_frame, ain_image_normalization_factor
        )

    perform_full_analysis = segmentation["perform_full_analysis"]

    # create pseudo_flat_field_corrected ocv_8-bit image from ocv_16-bit image
    ref_frame_8bit_flat_normalized = delayed(get_pseudo_opencv_8bit_flat_image)(
        ref_frame,
        ain_automatic_image_normalization,
        ain_image_normalization_factor,
        aic_apply_intensity_correction,
        aic_sigma,
    )

    img_binary = delayed(get_binary_image)(
        ref_frame_8bit_flat_normalized,
        segmentation["gb_ksize"],
        segmentation["gb_sigma"],
        segmentation["adapThresh_blockSize"],
        segmentation["adapThresh_constant"],
    )  # <---------------------------------------------- change here

    labels = delayed(apply_wShed_and_get_cluster_labels)(
        ref_frame_8bit_flat_normalized,
        img_binary,
        segmentation["min_distance"],
        segmentation["gb_ksize"],
        segmentation["gb_sigma"],
        segmentation["adapThresh_blockSize"],
        segmentation["adapThresh_constant"],
    )  # <-------------------- change here

    mask_img = delayed(create_mask_image)(labels)

    out_mask_each_cell = output_path / "mask_each_cell"
    out_mask_each_cell.mkdir(parents=True, exist_ok=True)

    # An image where contour lines are overlaid on detected cells (visualization purpose only)
    # ref_frame_8bit_flat_normalized_copy = ref_frame_8bit_flat_normalized.copy()
    ref_frame_8bit_flat_normalized_copy = delayed(cv2.cvtColor)(
        ref_frame_8bit_flat_normalized, cv2.COLOR_GRAY2BGR
    )

    t_final = delayed(create_t_final)(
        labels,
        ref_frame,
        ds_q.data,
        n_buff,
        out_mask_each_cell,
        perform_full_analysis,
        ref_frame_8bit_flat_normalized_copy,
    )

    # draft image: create a draft image and overlay detected objectes (visualisation only)
    draft_ref_frame_8bit_flat_normalized = delayed(
        create_draft_RGB_image_for_visualization
    )(ref_frame_8bit_flat_normalized, t_final)

    if not perform_full_analysis:
        ref_frame_8bit_flat_normalized_copy = delayed(stamp_image)(ref_frame_8bit_flat_normalized_copy, t_final)
        draft_ref_frame_8bit_flat_normalized = delayed(stamp_image)(draft_ref_frame_8bit_flat_normalized, t_final)

    return sink(
        ref_frame_8bit_flat_normalized,
        ref_frame_8bit_flat_normalized_copy,
        ref_frame_8bit,
        draft_ref_frame_8bit_flat_normalized,
        t_final,
        mask_img,
        input_path.stem,
        output_path,
    )


def map_uint16_to_uint8_skimage(img_16bit):
    """Converting 16-bit single channel image to 8-bit single channel image.
    Parameters
    ----------
    img_16bit : [numpy array]
        Single channel 16-bit image

    Returns
    -------
    [numpy array]
        Single channel 8-bit image
    """
    # Converting the input 16-bit image to uint8 dtype (using Scikit-Image)

    # reference:
    # http://scikit-image.org/docs/dev/user_guide/data_types.html

    # # library
    import warnings

    from skimage import img_as_ubyte

    # conversion
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img_8bit = img_as_ubyte(img_16bit)
    return img_8bit


def stamp_image(img, t_final):

    # legend properties
    REC_top_left = (100, 100)
    REC_w_h = (1000, 350)
    REC_color_in = (0, 0, 0)
    REC_color_out = (0, 255, 0)

    TXT1_loc = (175, 210)
    TXT1_color = (255, 255, 255)
    TXT1_size = 3.0
    TXT1_thickness = 4
    TXT1 = "Quick View Mode"

    TXT2_loc = (175, 290)
    TXT2_color = (255, 255, 255)
    TXT2_size = 1.5
    TXT2_thickness = 2
    TXT2 = "Number of detections: " + str(len(t_final))

    img = cv2.rectangle(img, REC_top_left, REC_w_h, REC_color_in, -1)
    img = cv2.rectangle(img, REC_top_left, REC_w_h, REC_color_out, 1)
    img = cv2.putText(
        img,
        TXT1,
        TXT1_loc,
        cv2.FONT_HERSHEY_SIMPLEX,
        TXT1_size,
        TXT1_color,
        TXT1_thickness,
    )
    img = cv2.putText(
        img,
        TXT2,
        TXT2_loc,
        cv2.FONT_HERSHEY_SIMPLEX,
        TXT2_size,
        TXT2_color,
        TXT2_thickness,
    )
    return img


def create_draft_RGB_image_for_visualization(imgOpencv_8bit, t_final):
    """
    Create a draft image where detected objects in the catalogue are overlaid for visualisation purpose only.

    Parameters
    ----------
    img_8bit : [numpy array]
        Single channel 8-bit image
    t_final : [astropy table]
        Final object catalogue

    Returns
    -------
    [numpy array]
        A three channel BGR/8-bit image with detected objects overlaid on the input image

    """

    imgOpencv_8bit_copy = imgOpencv_8bit.copy()
    imgOpencv_8bit_copy = cv2.cvtColor(imgOpencv_8bit_copy, cv2.COLOR_GRAY2BGR)
    # imgOpencv_8bit_copy = cv2.applyColorMap(imgOpencv_8bit_copy, cv2.COLORMAP_JET)
    # imgOpencv_8bit_copy = cv2.GaussianBlur(imgOpencv_8bit_copy, (3, 3), 0)
    # imgOpencv_8bit_copy = cv2.GaussianBlur(imgOpencv_8bit_copy,(0,0),sigmaX = 0.5, sigmaY = 0.5)

    # (ii) overlay position of detected object on this image
    for rows in t_final:
        cv2.circle(
            imgOpencv_8bit_copy,
            (int(rows["X_image"]), int(rows["Y_image"])),
            2,
            (0, 0, 255),
            -1,
        )
        cv2.circle(
            imgOpencv_8bit_copy,
            (int(rows["X_image"]), int(rows["Y_image"])),
            1,
            (0, 255, 255),
            -1,
        )

    return imgOpencv_8bit_copy
