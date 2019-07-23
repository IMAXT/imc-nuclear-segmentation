import logging
import traceback
from pathlib import Path

from dask import delayed
from distributed import Client, as_completed

from imc_pipeline import imcutil

from .preprocess import preprocess

log = logging.getLogger('owl.daemon.pipeline')


def main(
    n_buff: int = None,
    normalized_factor: int = None,
    img_path=None,
    output_path=None,
    segmentation=None,
):
    """IMC pipeline.

    Parameters
    ----------
    n_buff
        Number of pixels expaded around the segmented cell
        (excluding the cell itself)
    normalized_factor
        Pixel intensity normalization factor
    img_path : [type], optional
        [description], by default None
    output_path : [type], optional
        [description], by default None
    segmentation : [type], optional
        [description], by default None

    Raises
    ------
    FileNotFoundError
        [description]
    """
    # TODO: Complete the docstring

    client = Client.current()

    log.info('Starting IMC pipeline.')

    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    img_list = preprocess(img_path, output_path / 'cubes')

    futures = []
    for img_file in img_list:
        res = delayed(imcutil.process_image)(
            img_file, n_buff, normalized_factor, segmentation, output_path
        )
        fut = client.compute(res)
        futures.append(fut)

    for fut in as_completed(futures):
        if not fut.exception():
            log.info(fut.result())
        else:
            log.error(fut.exception())
            tb = fut.traceback()
            log.error(traceback.format_tb(tb))

    log.info('IMC pipeline finished.')
