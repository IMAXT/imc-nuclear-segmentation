"""

This function, reads output from IMC data packager and convert it to data cube

input data : OME TIFF Arrays (one array per IMC antibody channel - float32)
output data: 16-bit TIFF Image cube (one Image for all channles associated to the same IMC slice)

"""

import glob
import os

import numpy as np
from numpy import inf
from PIL import Image

import tifffile

# input parameters
# ================
imgPath_ome = 'DATA02__3D_stpt_IMC_DataPackager'
imgPath_ome_key = 'Q001'
imgFormat_ome = '.ome.tif'
imgFormat_cube = '.tiff'


# Creating CUBE TIFF files
# ========================

# OME-TIF folder names
imgFolderName_ome = [
    name
    for name in os.listdir(imgPath_ome)
    if os.path.isdir(os.path.join(imgPath_ome, name))
]

# expected name of CUBE TIFF images
imgFileName_cube_expected = [s + imgFormat_cube for s in imgFolderName_ome]

# name of current CUBE TIFF images
imgFileName_cube_current = [
    entry.name for entry in os.scandir(imgPath_ome) if entry.is_file()
]

# difference between current CUBE files and those expected
imgFileName_cube_missed = list(
    set(imgFileName_cube_expected) - set(imgFileName_cube_current)
)

# current content of CUBE image files
imgList_cube_version = glob.glob(imgPath_ome + '/*' + imgFormat_cube)


# check if all expected tiff images exist
if len(imgFileName_cube_missed) != 0:

    # prompt on terminal
    print('\n Creating CUBE image files  in ' + imgPath_ome + ' ... \n')

    # location of OME.TIF slices associated for each CUBE file
    imgPath_to_ome_files_with_missed_CUBE_images = [
        os.path.splitext(x)[0] + '/' + imgPath_ome_key + '/'
        for x in imgFileName_cube_missed
    ]

    # Loop through all OME.TIF files and construct CUBE image files
    for imgCUBE_index in range(len(imgPath_to_ome_files_with_missed_CUBE_images)):

        # Create a list of CUBE individual frames
        imgCUBE_frames = []

        # create individual TIFF cubes from OME.TIF
        imgFrames_ome = glob.glob(
            imgPath_ome
            + '/'
            + imgPath_to_ome_files_with_missed_CUBE_images[imgCUBE_index]
            + '*'
            + imgFormat_ome
        )
        imgFrames_ome.sort()

        for n_ch, frame in enumerate(imgFrames_ome):

            # read individual ome.tif frame
            img_ome_float32 = tifffile.imread(frame)

            # convert to 16-bit image
            img_ome_uint16 = img_ome_float32.astype(np.uint16)

            # remove inf values (pixels originally greter than 2**16 OR real 32-bit pixel values)
            img_ome_uint16[img_ome_uint16 == inf] = 0

            # convert numpy array to PIL array and add frames to the CUBE frame list
            imgCUBE_frames.append(Image.fromarray(img_ome_uint16))

        # Creating the final CUBE file (no-compression)
        imgCUBE_frames[0].save(
            imgPath_ome + '/' + imgFileName_cube_missed[imgCUBE_index],
            append_images=imgCUBE_frames[1:],
            save_all=True,
        )

else:
    print('\n Data CUBE already exist in \n ' + imgPath_ome)
