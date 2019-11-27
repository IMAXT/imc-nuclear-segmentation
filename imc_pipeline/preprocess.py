import logging
import re
from pathlib import Path
from typing import List

import xarray as xr

from imaxt_image.io import TiffImage

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

        for cube in slide.glob('Q???'):
            channels = []
            for p in cube.glob('*.tif'):
                with TiffImage(p) as img:
                    shape = img.shape
                    dimg = img.to_dask()
                ch = int(re.compile(r'Ch(\d\d\d)').search(p.name).groups()[0])
                arr = xr.DataArray(
                    dimg[None, :, :],
                    name=cube.name,
                    dims=['channel', 'y', 'x'],
                    coords={
                        'channel': [ch],
                        'x': range(shape[1]),
                        'y': range(shape[0]),
                    },
                )
                channels.append(arr)
            channels = xr.concat(channels, dim='channel')
            ds = channels.to_dataset()
            out = f'{slide.name}-{cube.name}.zarr'
            ds.to_zarr(output_dir / out, mode='w')
            print(out)
            filelist.append(output_dir / out)
    return filelist
