import logging
import traceback
# from pathlib import Path

from dask import delayed
from distributed import Client, as_completed

from imc_pipeline import imcutil

# from .preprocess import preprocess
from .preprocess_NEW import preprocess

log = logging.getLogger('owl.daemon.pipeline')


def main(
    n_buff: int = None,
    input_path=None,
    output_path=None,
    segmentation=None,
):
    """IMC pipeline.

    Parameters
    ----------
    n_buff
        Number of pixels expaded around the segmented cell
        (excluding the cell itself)
    img_path : [str], optional
        Path to input IMC image files. This is where you keep IMC images that you want to analyze.
    output_path : [str], optional
        # Path to IMC pipeline output products (results of analysis are recorded here)
    segmentation : [type], optional
        [description], by default None

    Raises
    ------
    FileNotFoundError
        [description]
    """
    # TODO: Complete the docstring

    imcutil.validate_input_params(n_buff, input_path, output_path, segmentation)

    client = Client.current()

    log.info('Starting IMC pipeline.')

    # img_path = Path(img_path)
    # if not img_path.exists():
    #     raise FileNotFoundError(img_path)
    #
    # output_path = Path(output_path)
    # output_path.mkdir(parents=True, exist_ok=True)

    # img_list = preprocess(img_path, output_path / 'cubes')
    img_list, img_path = preprocess(input_path, output_path)

    futures = []
    # for img_file in img_list:
    #     res = delayed(imcutil.process_image)(
    #         img_file, n_buff, segmentation, output_path
    #     )
    for img_index , img_file in enumerate(img_list):
        res = delayed(imcutil.process_image)(
            img_file, n_buff, segmentation, img_path[img_index]
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
