# general
import glob
import os
import time

import yaml
from dask import delayed

# paralellization
from distributed import Client, LocalCluster, as_completed

import imc_packages as imcutil


def main(
    ref_channel=None,
    normalized_factor=None,
    imgFormat=None,
    imgFormatOut=None,
    catFormat=None,
    imgPath=None,
    output_key=None,
    output_key_ref=None,
    output_key_mask=None,
    output_key_cat=None,
    n_workers_param=None,
    memory_limit_param=None,
):
    """IMC segmentation
    
    Parameters
    ----------
    ref_channel
        Reference channel
    normalized_factor
        Normalization facator
    """
    # time
    os.system('clear')

    start_total = time.time()

    print('\n\t ** START ** \n')

    # # i/o parameters
    # imgFormat = '.tiff'
    # imgFormatOut = '.jpg'
    # catFormat = '.fits'
    # imgPath = 'DATA02__3D_stpt_IMC' #'DATA03__Dimitra_IMC_data' # 'test_image_IMC' 'DATA02__3D_stpt_IMC' #'DATA01__bodenmiller_data'

    imgList = glob.glob(imgPath + '/*' + imgFormat)

    # subject to the sample to be analyzed
    # ref_channel = 37       # Dimitri data: 37 ; IMC: 25 # from investigation done using FIJI
    # normalized_factor = 10 # Dimitri date: 10 ; IMC: 30 # try and error

    # location of output products
    outputPath = imgPath + output_key  # "/output/"
    outputPath_ref = outputPath + output_key_ref  # "/reference/"
    outputPath_mask = outputPath + output_key_mask  # "/mask/"
    outputPath_cat = outputPath + output_key_cat  # "/catalog/"

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    if not os.path.exists(outputPath_ref):
        os.makedirs(outputPath_ref)
    if not os.path.exists(outputPath_mask):
        os.makedirs(outputPath_mask)
    if not os.path.exists(outputPath_cat):
        os.makedirs(outputPath_cat)

    with LocalCluster(
        processes=True, n_workers=n_workers_param, memory_limit=memory_limit_param
    ) as cluster:  # use to minotir http://127.0.0.1:8787/status
        # with LocalCluster(processes=True, n_workers = 20, memory_limit = '10GB') as cluster: # use to minotir http://127.0.0.1:8787/status
        # with LocalCluster(processes=True, diagnostics_port=8787) as cluster:
        with Client(cluster) as client:
            futures = []
            for img_file in imgList:
                res = delayed(imcutil.process_image)(
                    img_file,
                    ref_channel,
                    normalized_factor,
                    outputPath_ref,
                    outputPath_mask,
                    outputPath_cat,
                    imgFormat,
                    imgFormatOut,
                    catFormat,
                )
                fut = client.compute(res)
                futures.append(fut)

            for fut in as_completed(futures):
                print(fut.result())

    # report total time
    end = time.time()
    print('\t Total processing time %.3f s\n' % ((end - start_total)))

    print('\n\t ** FINISH ** \n')


# -----------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    config = yaml.safe_load(open('imc_config.yaml'))
    print(config)
    main(**config)
