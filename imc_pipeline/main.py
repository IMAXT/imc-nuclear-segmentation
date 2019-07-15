import logging
from pathlib import Path

from dask import delayed
from distributed import Client, as_completed

from imc_pipeline import imcutil

from .preprocess import preprocess

log = logging.getLogger('owl.daemon.pipeline')


def main(
    ref_channel=None,
    n_buff=None,
    normalized_factor=None,
    img_format=None,
    img_format_out=None,
    cat_format=None,
    img_path=None,
    output_key=None,
    output_key_ref=None,
    output_key_mask=None,
    output_key_cat=None,
):
    """IMC segmentation pipeline

    Parameters
    ----------
    ref_channel : [type], optional
        [description] (the default is None, which [default_description])
    normalized_factor : [type], optional
        [description] (the default is None, which [default_description])
    imgFormat : [type], optional
        [description] (the default is None, which [default_description])
    imgFormatOut : [type], optional
        [description] (the default is None, which [default_description])
    catFormat : [type], optional
        [description] (the default is None, which [default_description])
    imgPath : [type], optional
        [description] (the default is None, which [default_description])
    output_key : [type], optional
        [description] (the default is None, which [default_description])
    output_key_ref : [type], optional
        [description] (the default is None, which [default_description])
    output_key_mask : [type], optional
        [description] (the default is None, which [default_description])
    output_key_cat : [type], optional
        [description] (the default is None, which [default_description])
    """
    # TODO: Complete the docstring

    client = Client.current()

    log.info('Starting IMC pipeline.')

    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    img_list = img_path.glob(f'*.{img_format}')

    # location of output products
    output_path = img_path / output_key
    output_path_ref = output_path / output_key_ref
    output_path_mask = output_path / output_key_mask
    output_path_cat = output_path / output_key_cat

    output_path.mkdir(exist_ok=True)

    img_list = preprocess(img_path, output_path / 'cubes')

    futures = []
    for img_file in img_list:
        res = delayed(imcutil.process_image)(
            img_file,
            ref_channel,
            n_buff,
            normalized_factor,
            output_path_ref,
            output_path_mask,
            output_path_cat,
            img_format,
            img_format_out,
            cat_format,
        )
        fut = client.compute(res)
        futures.append(fut)

    for fut in as_completed(futures):
        print(fut.result())

    log.info('IMC pipeline finished.')
