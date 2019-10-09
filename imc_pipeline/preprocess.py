import logging
from pathlib import Path
from typing import List

import dask.array as da
import numpy as np
import xarray as xr

from imaxt_image.external import tifffile as tf
from imaxt_image.image import TiffImage

log = logging.getLogger('owl.daemon.pipeline')


def preprocess(input_dir: Path, output_dir: Path) -> List[Path]:
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
    if not output_dir.exists():
        output_dir.mkdir()

    filelist = []
    # TODO: This can be run in parallel for each slice
    for slide in input_dir.glob('*'):
        if not slide.is_dir():
            continue
        sections = []
        for cube in slide.glob('Q???'):
            with TiffImage(cube) as img:
                shape = img.shape
                dimg = img.to_dask()
                sections.append(dimg.astype('uint16'))
        stack = da.stack(sections)
        arr = xr.DataArray(
            stack,
            name=cube.name,
            dims=['section', 'y', 'x'],
            coords={
                'section': range(len(sections)),
                'x': range(shape[1]),
                'y': range(shape[0]),
            },
        )
        ds = xr.Dataset()
        ds[cube.name] = arr
        output = output_dir / f'{slide.name}-{cube.name}.zarr'
        ds.to_zarr(output, mode='w')
        filelist.append(output)
    return filelist


"""
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
"""
