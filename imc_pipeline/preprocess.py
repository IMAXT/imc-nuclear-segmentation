import logging
from typing import List

from pathlib import Path
from imaxt_image.external import tifffile as tf
from imaxt_image.io import TiffImage

import numpy as np
import xarray as xr
import os

log = logging.getLogger("owl.daemon.pipeline")


TIFF_SUFFIX = ["tif", "tiff"]


def find_image_extension(dir: Path) -> str:
    """Loop through a directory and return the suffix of the first file match.

    Parameters
    ----------
    dir
        directory path
    suffixes
        list of suffixes to search for

    Returns
    -------
    suffix of first file that matches
    """
    for suffix in TIFF_SUFFIX:
        n_files = list(dir.glob(f"*.{suffix}"))
        if n_files:
            return suffix


# zarr reader (new)
def check_zarr(input_path: Path):
    return (input_path / '.zgroup').exists()


def read_individual_zarr_xarray_as_a_list(input_path: Path):
    ds = xr.open_zarr(f"{zarr_file}")
    q_ids = ds.attrs['meta'][0]['acquisitions']
    ds_q_all = [xr.open_zarr(f"{zarr_file}", group=q_id) for q_id in q_ids]
    ds_q = [ds for ds in ds_q if ds.attrs['meta'][0]['q_data_source']!="invalid"]
    return ds_q


def create_zarr_output_folders(input_path: Path, output_path: Path, q_ids: List):
    # setting output forlder names
    output_folderName = input_path.stem
    output_folderName_Pathlist = list()

    output_CUBEfileName = output_folderName + '_CUBE.tif'
    output_CUBEfileName_Pathlist = list()
    # creating output folders for each acquisition (roi)
    for each_roi in q_ids:
        output_folder_Path = Path(output_path, output_folderName, each_roi, 'CUBE_image')
        output_folderName_Pathlist.append(output_folder_Path)
        output_folder_Path.mkdir(parents=True, exist_ok=True)
        output_CUBEfileName_Pathlist.append(Path(output_folder_Path, output_CUBEfileName))

    return output_CUBEfileName_Pathlist


def create_master_zarr_data_cube_in_numpy_form(q_ids, ds_q):

    # on terminal display
    print('\t\t ----------------------------------------------------------- ')
    print('\t\t| STEP . 3: Creating master CUBE image data as NUMPY arrays |')
    print('\t\t ----------------------------------------------------------- ')

    q_ids_dataCube_masterList = list()
    q_ids_channelNameList = list()

    for acq_idx in range(len(q_ids)):

        print('\nProcessing:')
        print('\nacq_idx:', acq_idx, '\tq_ids:', q_ids[acq_idx])

        # list of channel names for a sepecific acquisition
        q_ids_channelName = ds_q[acq_idx].attrs['meta'][0]['q_channels']
        q_ids_channelNameList.append(q_ids_channelName)

        # # save list of channels as a text file
        # q_ids_channelNameList_fileName = str(Path(output_path, output_folderName, q_ids[acq_idx], 'CUBE.txt'))
        # np.savetxt(q_ids_channelNameList_fileName, q_ids_channelNameList, delimiter="\n", fmt="%s")
        # print('output path:\n', Path(output_path, output_folderName, q_ids[acq_idx], 'CUBE_image'))

        q_ids_dataCube_list = list()
        for ch in range(len(q_ids_channelName)):
            ch_data = np.array(ds_q[acq_idx][q_ids[acq_idx]][ch])  # convert to numpy float32
            ch_data = ch_data.astype(np.uint16)  # float32 -> uint-16bit
            q_ids_dataCube_list.append(ch_data)

        # Cast the list into a numpy array
        q_ids_dataCube_array = np.array(q_ids_dataCube_list)  # uint16
        q_ids_dataCube_array.shape

        q_ids_dataCube_masterList.append(q_ids_dataCube_array)

    # Convert two lists to a dictionary
    # q_ids -> list of names of acquisitions
    # q_ids_dataCube_masterList -> list of cube_arrays for each acquisition
    # ----------------------------------------------------------------------

    # Create a zip object from two lists
    q_ids_zipbObj = zip(q_ids, q_ids_dataCube_masterList)

    # Create a dictionary from zip object
    q_ids_masterDict = dict(q_ids_zipbObj)

    return q_ids_masterDict, q_ids_channelNameList


def save_master_zarr_data_cube_as_big_tiff(q_ids_masterDict, q_ids_channelNameList, output_CUBEfileName_Pathlist):

    # on terminal display
    print('\t\t ---------------------------------------------------------------------- ')
    print('\t\t| STEP . 4: Saving master CUBE image data as BIG TIFF data CUBE arrays |')
    print('\t\t ---------------------------------------------------------------------- ')

    q_ids = list(q_ids_masterDict.keys())

    print('Saving ... \n')

    for acq_idx in range(len(q_ids)):

        print('\t -- Acquisition ID: ', q_ids[acq_idx])

        # i/o: path + filename
        # cube_filename = str( Path(output_path, output_folderName, q_ids[acq_idx], 'CUBE_image', output_CUBEfileName))
        cube_filename = str(output_CUBEfileName_Pathlist[acq_idx])

        # specific cube array
        channel_cube_array = q_ids_masterDict[q_ids[acq_idx]]

        # Saving Channel Names
        # Also saving channel names for each acquisition inside acquisition folder
        channel_names = q_ids_channelNameList[acq_idx]
        channel_name_filename = 'CUBE.txt'
        channel_name_filename = cube_filename.split('CUBE_image')[0] + channel_name_filename
        np.savetxt(channel_name_filename, channel_names, delimiter="\n", fmt="%s")
        print('\t --', channel_name_filename)

        # write as big tif
        with tf.TiffWriter(cube_filename, bigtiff=True) as tif:
            for i in range(channel_cube_array.shape[0]):
                tif.save(channel_cube_array[i], compress=0)
        print('\t --', cube_filename, '\n')
    pass


# zarr reader (new)
def create_cube_zarr(input_path: Path, output_path: Path):

    log.info("Creating Zarr cube %s, %s", input_path, output_path)

    ds_q = read_individual_zarr_xarray_as_a_list(input_path)

    #  S T E P . 2: Creating output folders and get a list of unique filenames (output path + full name) for each ROI (acquisition)
    output_CUBEfileName_Pathlist = create_zarr_output_folders(input_path, output_path, q_ids)
    print("\n Unique file-names for ", q_ids , " are : ", output_CUBEfileName_Pathlist)

    #  S T E P . 3: Creating a master dictionary of ROIs aka acquisitions, and their corresponding CUBE image data as numpy arrays
    q_ids_masterDict , q_ids_channelNameList = create_master_zarr_data_cube_in_numpy_form(q_ids, ds_q)

    #  S T E P . 4: Saving master CUBE image data as BIG TIFF data CUBE arrays
    save_master_zarr_data_cube_as_big_tiff(q_ids_masterDict, q_ids_channelNameList, output_CUBEfileName_Pathlist)

    #  S T E P . 5: Return img_list & img_path
    # list of full path to each CUBE image file created
    img_list = output_CUBEfileName_Pathlist

    # from 'img_list' (see above), I extract only the path to each acquisition (.../.../up_to_/Q00$)
    # and append them to a list as Path objects
    img_path = list()
    for tiff_cube in img_list:
        path_tiff_cube = str(tiff_cube).split('CUBE_image')[0]
        img_path.append(Path(path_tiff_cube))

    return img_list, img_path


def check_ometif(input_path: Path) -> List:

    # see if ome-tif unique keys exists among subfolders (and if they are repeated)
    found_roi_unique_keys = [*input_path.glob("Q???")]

    # if ome-tif subfolder does not exist, check if TIF images are within current folder
    if not found_roi_unique_keys:
        # check if image files are TIF or TIFF
        suffix = find_image_extension(input_path)
        if not suffix:
            raise FileNotFoundError(
                "The input path does not have any tif image files. Please check the input path again"
            )
        data_is_ometif = False
    else:
        md5_files = [*input_path.rglob("*.md5")]
        tif_files = [*input_path.rglob("*.ome.tif")]
        if not tif_files:
            raise FileNotFoundError(
                "Incorrect OME.TIF directory: no tiff files present in '%s'", input_path
            )
        elif not md5_files:
            raise FileNotFoundError(
                "Incorrect OME.TIF directory: no md5 files present in '%s'", input_path
            )
        elif len(md5_files) != len(tif_files):
            raise Exception("Incorrect number of tif and md5 files in '%s'", input_path)

        suffix = "tif"
        data_is_ometif = True

    return data_is_ometif, suffix


def create_cube_normal(
    input_path: Path, output_path: Path, tif_ext: str = "tif", compress: int = 0
) -> List:
    image_files = sorted(input_path.glob(f"*.{tif_ext}"))
    channel_names = [filename.name.split(".")[0] for filename in image_files]

    cube_path = output_path / input_path.name / "CUBE_image"
    cube_path.mkdir(parents=True, exist_ok=True)

    cube_fullname = cube_path / f"{input_path.name}_CUBE.tif"
    cube_txt = output_path / input_path.name / "CUBE.txt"

    if not cube_fullname.exists():
        # keep record of channel names (= input tif filenames)
        with tf.TiffWriter(cube_fullname, bigtiff=True) as tif:
            with open(cube_txt, "w") as fh:
                # loop through tif files
                for indx, frame in enumerate(image_files):

                    # read individual 16-bit tif image
                    img = tf.imread(f"{frame}").astype("uint16")
                    metadata = {f"{channel_names[indx]}": indx + 1}

                    # write channel names on an output file
                    fh.write(f"{indx+1},{channel_names[indx]}\n")

                    tif.save(
                        img, compress=compress, metadata=metadata,
                    )
    else:
        log.info("CUBE data file already exists '%s'", cube_fullname)

    return [cube_fullname], [output_path]


def create_cube_ome(
    input_path: Path, output_path: Path, tif_ext: str = "tif", compress: int = 0
) -> List:
    img_list = []
    img_path = []
    for roi in input_path.glob("Q???"):
        log.debug(f"Processing {roi}")
        image_files = sorted(roi.glob(f"*.{tif_ext}"))

        cube_path = output_path / input_path.name / roi.name / "CUBE_image"
        cube_path.mkdir(parents=True, exist_ok=True)

        cube_fullname = cube_path / f"{input_path.name}_CUBE.tif"
        cube_txt = output_path / input_path.name / roi.name / "CUBE.txt"

        # add Path object (version) of image name and its path
        img_list.append(cube_fullname)
        img_path.append(output_path / input_path.name / roi.name)

        if not cube_fullname.exists():
            with tf.TiffWriter(cube_fullname, bigtiff=True) as tif:
                with open(cube_txt, "w") as fh:
                    # loop through tif files
                    for indx, frame in enumerate(image_files):

                        # read individual 16-bit tif image
                        img = TiffImage(frame)
                        arr = img.asarray().astype("uint16")
                        metadata = img.metadata.as_dict()
                        ch_name = metadata["OME"]["Image"]["Pixels"]["Channel"][
                            "@Fluor"
                        ]

                        # write channel names on an output file
                        fh.write(f"{indx+1},{ch_name}\n")

                        tif.save(
                            arr, compress=compress, metadata=metadata,
                        )
        else:
            log.info("CUBE data file already exists '%s'", cube_fullname)

    return img_list, img_path


def preprocess(input_path: Path, output_path: Path, compress=0) -> List[Path]:
    """Run preprocessing of input images.

    The function reads input image path 'img_path' and analyze its contents in
    order to check if it contains normal TIFF files (where each file is an IMC
    channel belonging to the same IMC run) or a OME-TIF file. In the first case
    it is always a single ROI IMC run. In the second case it can vbe a single or
    multiple IMC run.

    Parameters
    ----------
    input_path:
       Path to the input folder associated to an IMC run
    output_path:
       Path to where output products will be stored.

    Returns
    -------
    list contining list of images and output path
    """

    if not isinstance(input_path, Path):
        input_path = Path(input_path)

    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    # first check to see if input data is of ZARR format
    data_is_zarr = check_zarr(input_path)

    # If ZARR format
    if data_is_zarr:
        img_list, img_path = create_cube_zarr(input_path, output_path)

    # if *NOT* ZARR format: Check if it is OME or NORMAL format
    else:
        data_is_ometif, tif_ext = check_ometif(input_path)

        if not data_is_ometif:
            img_list, img_path = create_cube_normal(input_path, output_path)
        else:
            img_list, img_path = create_cube_ome(input_path, output_path)

    return img_list, img_path
