import logging
from pathlib import Path
from typing import List

import numpy as np

from imaxt_image.external import tifffile as tf
from imaxt_image.image import TiffImage

log = logging.getLogger('owl.daemon.pipeline')


def preprocess(input_dir: Path, output_dir: Path) -> List[Path]:
    """ Converts OME.TIFF images (associated with individual IMC channels of the same slice) into a single cube TIFF image: The IMC pipeline reads image cubes i.e. a single TIFF image file, containing all IMC image channels for the same slice. If the format of the input image is OME.TIFF (which is the data packager format), then this function convert that format into cube TIFF format and returns a list of the name/location of converted images.

    Parameters
    ----------
    input_dir
        Input directory containing OME.TIFF images
    output_dir
        Directory where TIFF cubes are written

    Returns
    -------
    list of filenames (one per image cube)
    """
    if not output_dir.exists():
        output_dir.mkdir()

    filelist = []
    # TODO: This can be run in parallel for each slice
    for slide in input_dir.glob('*'):
        if not slide.is_dir():
            continue
        for cube in slide.glob('Q???'):
            output = output_dir / f'{slide.name}-{cube.name}.tif'
            if output.exists():
                log.debug('%s already exists', output)
                filelist.append(output)
                continue

            imgs = [TiffImage(im).asarray() for im in sorted(cube.glob('*.tif'))]
            imgs = np.stack(imgs).astype('uint16')
            imgs[imgs == np.inf] = 0

            try:
                out = tf.TiffWriter(output)
                out.save(imgs)
                log.info('%s saved', output)
                filelist.append(output)
            except Exception:
                log.critical('Cannot save file %s', output)
                if output.exists():
                    output.unlink()
    return filelist
