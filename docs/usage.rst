Usage
=====


.. _imc_pipedef:

Pipeline definition file
-------------------------

In order to run a pipeline we need a configuration file that sets the inputs,
outputs and relevant paramenters needed for the various steps.

The following command:

.. code-block:: python

    python -m imc_pipeline config

will display a default configuration file similar to:

.. code-block:: yaml

    version: 1

    name: imc
    extra_pip_packages: imc-pipeline

    img_path: /data/meds1_a/imaxtapp/incoming/MA1-0002/IMC	# Path to files associated with an IMC run. The path should contain either (a) a series of TIF files (*tif *tiff), where each file name is associated with an IMC channel and files together belong to an IMC run (e.g. 23 files if there are 23 channels) or (b) Q00X (x=1, 2, ...) folders.   
    output_path: /data/meds1_b/imaxt/imc/MA1-0002		# Path to IMC pipeline output products (results of analysis are recorded here)

    n_buff: 1							# [pixels ; Recommended 0 < n_buff < 5 ] This is the width of periphery (or thickness of the edge) around each detected nucleus within which, the pipeline estimates the mean value of pixel intensities. If set to zero (=0), the pipeline does not measure any pixel intensity within the edges of detected nuclei. If too large e.g. > 5 [pixels], then there is a risk that the periphery is merged with peripheries of nearby cells (unless the cell is located in an isolated area) 


    # segmentation-related parameters
    segmentation:
	perform_full_analysis: False 				# if False, only a draft image is produced with detected cells overlaid on the reference (nuclear) channel
        ref_channel: 25 					# The IMC channel to be used for segmentation. This *should* be one of nuclear channels (check a sample image manually in imageJ or FIJI)
        min_distance: 3 					# [pixels; Watershed segmentation] Smaller values, tends to oversegmentation (finding too many cells). 
        gb_ksize: 0 						# [Denoising, 0, 3, 5, 7, and so on.] Gaussian blur kernel size - This cause some of the background noise to be removed before watershed segmentation.
        gb_sigma: 2.0 						# [Denoising] Gaussian blur sigma - This cause some of the background noise to be removed before watershed segmentation.
        adapThresh_blockSize: 15 				# [Image binarization; Odd integer] Size of a pixel neighborhood (Kernel) that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on. As a rule of thumb, it should be always greater than the largest possible cell diameter observed in the current IMC sample.
        adapThresh_constant: -7.5 				# [Image binarization; float(recommended < 0)] Constant subtracted from the mean or weighted mean (positive, zero or negative). But it is recommended to use negative values (meaning bright cells in dark background)
	ain_automatic_image_normalization: True 		# [boolean] - If TRUE, the pipeline automatically enhances the contrast of the input reference channel. If set to 'False', then the pipeline looks at the 'ain_image_normalization_factor' to perform image enhancement.
	ain_image_normalization_factor: 1			# [*10^{-2} percent; Recommended 1 to 50] During the processing, the IMC pipeline converts 16-bit images into 8-bit and recalculates the pixel values of the image so the range is equal to the maximum range for the data type. However, to maximise the image contrast, some of the pixels are allowed to become saturated. Therefore, increasing this value increases the overall contrast. If set to 1, there would be no saturated pixels and no change in image contrast. Note that if 'ain_automatic_image_normalization' is set to TRUE, then this parameter has no effect and is skipped.
	aic_apply_intensity_correction: False			# [boolean] - If TRUE, then the algorithm try to create a reference image with uniform pixel intensities. Initially, the algorithm convolve the input image (single channel) with a Gaussian kernel of standard deviation 'aic_sigma' [in pixels ; see next parameter] and then divide the original image by the filtered one. If FALSE, nothing happens and the parameter 'aic_sigma' (see next parameter) is ignored. Further information about the Gaussian filter used can be found here https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.filters.gaussian_filter.html
	aic_sigma: 5						# [pixels] Standard deviation for Gaussian kernel. Valid only if  aic_apply_intensity_correction = True

    resources:
      workers: 6


Create a file with this output
(e.g. ``imc_pipeline.yml``) and
modify at least the input path ``img_path`` and the output ``output_path``.

.. note:: ``output_path`` must be a subdirectory in ``/data/meds1_b/imaxt/``

Submitting the pipeline
-----------------------

In order to submit the pipeline you will need an account in the 
`IMAXT Archive <https://imaxt.ast.cam.ac.uk/archive>`_.

Run an ssh session into the IMAXT login node and type the following:

**Setup environment**

.. code-block:: python

    conda activate imaxt

**Setup credentials**

.. code-block:: python

    owl api login

Use your archive username and password.

**Submit the pipeline**

.. code-block:: python

    owl pipeline submit --conf imc_pipeline.yaml

The command above returns a ``jobID`` number.

**Check status in the command line**

Pipeline status and log messages can be checked using:

.. code-block:: python

    owl pipeline status jobID

where ``jobID`` is the pipeline number.

**Check status in the archive**

Progress can be monitored from the Web at https://imaxt.ast.cam.ac.uk/archive/owl/
