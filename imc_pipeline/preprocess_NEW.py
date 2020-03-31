import glob
import os
import sys
import numpy as np
# from numpy import inf
from pathlib import Path
from skimage.external import tifffile
from imaxt_image.io import TiffImage


# Get a list of immidiate current sub-folders
def SubDirPath(path):
    '''Read path and returns list of current subfolders'''
    if not os.path.isdir(path):
        print(f'\nThe image path:\n {path}\n * does not exist * \nPlease try again. Programme exit...!')
        sys.exit()

    list_of_subFolders_full = list(filter(os.path.isdir, [os.path.join(path, f) for f in os.listdir(path)]))   # subfolder full path
    list_of_subFolders = [subfold.split("/")[-1] for subfold in list_of_subFolders_full]                        # subfolder name only
    return list_of_subFolders


# extract image file name only (to extract channel name for normal tif files)
def get_file_name(file_path):

    # get base name
    base = os.path.basename(file_path)

    # split base string
    file_name = base.split('.')[0]

    return file_name


# get key associated with a value in a dictionary
def getKey(dct, value):
    return [key for key in dct if (dct[key] == value)][0]


def preprocess(input_path, output_path):
    '''
    The function reads input image path 'img_path' and analyze its contents in order to check if it contains
       (a) Normal-TIFF files (where each file is an IMC channel belongs to the same IMC run), OR
       (b) OME-TIFF file.
       In case (a), it is always a single ROI IMC run.
       In case (b), it could be a 'single' or 'multiple' IMC run.

    Input:
       input_path [str]: Path to the input folder associated to an IMC run
       output_path [str]: Path to where output products will be stored.

    Output:
       img_list [Path object]: Path to image cube(s)
       img_path [Path object]: Path to output folder(s).
          - In case of Normal-TIFF, output folder path (img_path) is the same as inputted 'output_path'
          - In case of OME-TIFF, output folder(s) are different and within sub-folders inside the inputted 'output_path'
    '''

    # create a dictionary of in/out path
    path = {'in': input_path, 'out': output_path}

    # Correct the input path if it does not have '/' at the end
    for p in path:
        if path[p][-1] != '/':
            path[p] += '/'

    # list of ome.tif key folders
    ometif_roi_keyword = ['Q001', 'Q002', 'Q003', 'Q004', 'Q005', 'Q006', 'Q007', 'Q008', 'Q009']
    ometif_sign = '.md5'
    n_imc_channel_per_ROI = []
    possible_tif_signs = ['.tif', '.tiff']
    tif_ext = False
    tif_compress = 0

    # final decision
    data_is_ometif = False
    data_is_normal_tif = False

    output_img_type = '.tiff'
    output_chname_type = '.txt'

    os.system('clear')

    '''
     -------------
    | P A R T . 1 |
     -------------
     DETERMINE INPUT TYPE: Normal TIF or OME.TIF ?
     if Normal TIF >>> data_is_normal_tif --> TRUE
     if OME TIF    >>> data_is_ometif     --> TRUE
    '''
    # get list of current sub-folders
    subfolders = SubDirPath(path['in'])

    # see if ome-tif unique keys exists among subfolders (and if they are repeated)
    found_roi_unique_keys = [i for i in subfolders if i in ometif_roi_keyword]

    # if ome-tif subfolder does not exist, check if TIF images are within current folder
    if not found_roi_unique_keys:
        print('\nThe input path does not have ome-tif file. Please check the input path agin if you expect one.')

        # check if image files are TIF or TIFF. Also check if total number of TIF and MD5 fles are the same
        for ext in possible_tif_signs:
            n_tif_files = len(glob.glob(path['in'] + '*' + ext))

            if n_tif_files > 0:
                print(f'\n *** The input path seem to have normal {ext} image files. ***')
                data_is_normal_tif = True
                tif_ext = ext

        if not data_is_normal_tif:
            print('\nThe input path does not have any tif image files. Please check the input path agin ... \nProgramme exit ...!')
            sys.exit()
    else:

        # check individual ROIs and check:
        # (a) if they all have "TIF + MD5" files and
        # (b) determine the TIF ext type (TIF or TIFF)
        # (c) check if all ROIs have the same number of channels
        for roi in found_roi_unique_keys:

            # check if MD5 files exist
            n_md5_files = len(glob.glob(path['in'] + roi + '/*' + ometif_sign))

            # if MD5 file exist, then...
            if n_md5_files != 0:

                n_imc_channel_per_ROI.append(n_md5_files)

                # check if image files are TIF or TIFF. Also check if total number of TIF and MD5 fles are the same
                for ext in possible_tif_signs:

                    n_tif_files = len(glob.glob(path['in'] + roi + '/*' + ext))
                    if n_tif_files == n_md5_files:

                        tif_ext = ext

                        print(f'{roi} contains {n_tif_files} image files with extention {tif_ext}')

                # if number of TIF and MD5 files are not the same (or there are not TIF images)
                if not tif_ext:

                    print(f'Mismatch number of {ometif_sign} and {ext}. \nProgramme exit ...!')
                    sys.exit()

            # MD5 does not exist
            else:
                print(f'No associated {ometif_sign} files. \nProgramme exit ...!')
                sys.exit()

        # finally check the number of channels in each ROI
        if len(set(n_imc_channel_per_ROI)) != 1:
            print('\n ROIs have different number of image channels. \nProgramme exit ...!')
            sys.exit()

        else:
            data_is_ometif = True
            print('\n ROIs seem to be fine.\n')

    '''
     -------------
    | P A R T . 2 |
     -------------
     CREATE IMAGE CUBE if >> Normal TIFF <<
    '''
    # final product
    # preprocess_product = {'imageFile' : list(),
    #                      'output_for_imageFile' : list()
    #                      }
    img_list = []
    img_path = []

    ###################################################
    # NORMAL MODE #######
    ###################################################

    if data_is_normal_tif:
        list_of_image_files = glob.glob(path['in'] + '*' + tif_ext)
        list_of_image_files.sort()
        list_of_channel_names = [get_file_name(ch_name) for ch_name in list_of_image_files]
        tif_file = {'img': list_of_image_files, 'channel': list_of_channel_names, 'cube': [], 'metadata': {}}

        # output
        img_cube = {'basename': path['in'].split('/')[-2],
                    'key': '_CUBE',
                    'format': output_img_type,
                    'path': path['out'] + 'CUBE_image/'}

        img_cube['fullname'] = img_cube['path'] + img_cube['basename'] + img_cube['key'] + img_cube['format']
        img_cube['channel_fileName'] = img_cube['path'] + img_cube['basename'] + img_cube['key'] + output_chname_type

        # add Path object (version) of image name and its path
        img_list.append(Path(img_cube['fullname']))
        img_path.append(Path(path['out']))

        # preprocess_product[ img_cube['fullname'] ] = path['out']
        # preprocess_product['imageFile'].append(img_cube['fullname'])
        # preprocess_product['output_for_imageFile'].append(path['out'])

        # create output folder
        if not os.path.exists(img_cube['path']):
            os.makedirs(img_cube['path'])

        if not os.path.exists(img_cube['fullname']):
            # keep record of channel names (= input tif filenames)
            file = open(img_cube['channel_fileName'], "w")

            # loop through tif files and read them
            for tif_index , tif_frame in enumerate(tif_file['img']):

                # read individual 16-bit tif image
                tif_file['cube'].append(tifffile.imread(tif_frame))                                        # add each frame to a lis
                tif_file['metadata'][tif_file['channel'][tif_index]] = (tif_index + 1)
                file.write(str(tif_index + 1) + ' , ' + tif_file['channel'][tif_index] + '\n')   # write channel names on an output file

            file.close()

            # write output image(cube) as a multi-frame / numpy array as big tiff file
            tif_file['cube'] = np.asarray(tif_file['cube'], dtype=np.uint16)                                    # convert imgCUBE_frames (list) to numpy array

            with tifffile.TiffWriter(img_cube['fullname'], bigtiff=True) as tif:
                for i in range(len(tif_file['channel'])):
                    tif.save(tif_file['cube'][i], compress=tif_compress, metadata=tif_file['metadata'])
        else:
            print('\n CUBE data file already exists: \n')
            print(img_cube['fullname'])
            pass

        '''
         --------------------
        | P A R T . 3 (below)|
         --------------------
        CREATE IMAGE CUBE if >> OME TIFF <<
        '''

        ###################################################
        # OME-TIF MODE #######
        ###################################################

    elif data_is_ometif:

        for _roi_index, roi_folder in enumerate(found_roi_unique_keys):

            print(f'\nReading {roi_folder}...')
            list_of_image_files = glob.glob(path['in'] + roi_folder + '/*' + tif_ext)
            list_of_image_files.sort()
            list_of_channel_names = [get_file_name(ch_name) for ch_name in list_of_image_files]

            # unlike normal-tif where channel names are filenames, here channel names must be read from the metadata
            tif_file = {'img': list_of_image_files, 'channel': [], 'cube': [], 'metadata': {}}

            img_cube = {'basename': path['in'].split('/')[-2] + '_' + roi_folder,
                        'key': '_CUBE',
                        'format': output_img_type,
                        'path': path['out'] + 'CUBE_image/'
                        }
            img_cube['fullname'] = img_cube['path'] + img_cube['basename'] + img_cube['key'] + img_cube['format']
            img_cube['channel_fileName'] = img_cube['path'] + img_cube['basename'] + img_cube['key'] + output_chname_type

            # add Path object (version) of image name and its path
            img_list.append(Path(img_cube['fullname']))
            img_path.append(Path(path['out'] + '/' + roi_folder + '/'))
            # preprocess_product[ img_cube['fullname'] ] = path['out']
            # preprocess_product['imageFile'].append(img_cube['fullname'])
            # preprocess_product['output_for_imageFile'].append(path['out']+ roi_folder)

            # create output folder
            if not os.path.exists(img_cube['path']):
                os.makedirs(img_cube['path'])

            if not os.path.exists(img_cube['fullname']):

                # keep record of channel names (= input tif filenames)
                file = open(img_cube['channel_fileName'], "w")

                # loop through tif files and read them
                for tif_index , tif_frame in enumerate(tif_file['img']):

                    # create img pointer from ome.tif
                    img = TiffImage(tif_frame)

                    # read meta data
                    metadata = img.metadata.as_dict()

                    # extract channel name associate with the ome.tif
                    ch_name = metadata['OME']['Image']['Pixels']['Channel']['@Fluor']  # ['@Name'] <-- problem: there are similar names

                    # create 16-bit numpy array from ome.tif image pointer
                    img_uint16 = img.asarray().astype(np.uint16)

                    # update tif cube dictionary
                    tif_file['cube'].append(img_uint16)
                    tif_file['metadata'][ch_name] = (tif_index + 1)
                    file.write(str(tif_index + 1) + ' , ' + ch_name + '\n')  # write channel names on an output file

                    print(str(tif_index + 1) + ' , ' + ch_name)

                with tifffile.TiffWriter(img_cube['fullname'], bigtiff=True, imagej=True) as tif:
                    for i in range(len(tif_file['metadata'])):
                        tif.save(tif_file['cube'][i], compress=tif_compress, metadata=tif_file['metadata'])

                file.close()

            else:
                print('\n CUBE data file already exists: \n')
                print(img_cube['fullname'])
                pass

    # return final product i.e. cube image locations + their associateed output path
    return img_list, img_path
