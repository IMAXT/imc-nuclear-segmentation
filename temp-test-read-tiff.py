from imaxt_image.image import TiffImage
import cv2
import numpy as np
from PIL import Image
from typing import List

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

# find number of channels
from skimage.external import tifffile
def get_frames_NEW(cube: Image) -> List[np.ndarray]:
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

img = '/home/darius01/myDisk1/myProjects/medical/imaxt/imc/OUTPUT_temp/tiff-single-ome-large-ref45_Q001_CUBE.tiff' 


# get_frames(img_small)  <-- IS EQUAL TO --> tifffile.imread(img_small)
# above does not work for large tiff files
