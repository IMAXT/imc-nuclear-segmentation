import logging
from typing import List

from pathlib import Path
from imaxt_image.external import tifffile as tf
from imaxt_image.io import TiffImage


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


def check_ometif(input_path: Path) -> List:
    """
    """
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

    data_is_ometif, tif_ext = check_ometif(input_path)

    if not data_is_ometif:
        img_list, img_path = create_cube_normal(input_path, output_path)
    else:
        img_list, img_path = create_cube_ome(input_path, output_path)

    return img_list, img_path
