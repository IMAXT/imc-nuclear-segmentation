import logging
from pathlib import Path

import numpy as np

from imaxt_image.external import tifffile as tf
from imaxt_image.image import TiffImage

log = logging.getLogger('owl.daemon.pipeline')


def preprocess(input_dir: Path, output_dir: Path):
    """Preprocess IMC input image directory.

    Individual images are saved as cubes.

    Parameters
    ----------
    input_dir
        Input directory containing images
    output_dir
        Directory where cubes are written

    Returns
    -------
    list of filenames, each containing a cube
    """
    filelist = []
    for slide in input_dir.glob('*'):
        if not slide.is_dir():
            continue
        for cube in slide.glob('Q???'):
            output = output_dir / f'{slide.name}-{cube.name}.tif'
            if output.exist():
                log.debug('%s already exists', output)
                filelist.append(output)
                continue

            imgs = [TiffImage(im).asarray() for im in cube.glob('*.tif')]
            imgs = np.stack(imgs).astype('uint16')
            imgs[imgs == np.inf] = 0

            try:
                out = tf.TiffWriter(output)
                out.save(imgs)
                log.info('%s saved', output)
                filelist.append(output)
            except Exception:
                log.critical('Cannot save file %s', output)
                if output.exist():
                    output.unlink()

    return filelist
