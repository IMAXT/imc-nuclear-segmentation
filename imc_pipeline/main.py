import logging
from pathlib import Path
from typing import Dict, Any

import xarray as xr

from imc_pipeline import imcutil


log = logging.getLogger('owl.daemon.pipeline')


def main(
    n_buff: int = None,
    input_path: Path = None,
    output_path: Path = None,
    segmentation: Dict[str, Any] = None,
):
    """IMC pipeline.

    Parameters
    ----------
    n_buff
        Number of pixels expaded around the segmented cell
        (excluding the cell itself)
    img_path
        Path to input IMC image files. This is where you keep IMC images that you want to analyze.
    output_path
        Path to IMC pipeline output products (results of analysis are recorded here)
    segmentation
        Segmentation configuration.

    Raises
    ------
    FileNotFoundError
        [description]
    """
    # TODO: Complete the docstring
    log.info('Starting IMC pipeline.')

    output_path.mkdir(parents=True, exist_ok=True)

    # img_list, img_path = preprocess(input_path, output_path)

    ds = xr.open_zarr(f"{input_path}")
    acqs = ds.attrs['meta'][0]['acquisitions']

    for acq in acqs:
        output = output_path / input_path.stem / acq
        res = imcutil.process_image(input_path, acq, n_buff, segmentation, output)
        res.compute()

    log.info('IMC pipeline finished.')
