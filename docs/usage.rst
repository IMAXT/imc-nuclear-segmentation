Usage
=====

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

    img_path: /data/meds1_a/imaxtapp/incoming/MA1-0002/IMC
    output_path: /data/meds1_b/imaxt/imc/MA1-0002

    # flux-estimation (mean pixel intensities) related parameters
    n_buff: 1

    # flatfield correction
    normalized_factor: 30

    # segmentation-related parameters
    segmentation:
        ref_channel: 25 # nuclear channel to be used for segmentation
        min_distance: 3 # watershed segmentation
        gb_ksize: 0 # gaussian blur kernel size (for denoising)
        gb_sigma: 2.0 # gaussian blur sigma (for denoising)
        adapThresh_blockSize: 15 # kernel size for background removal
        adapThresh_constant: -7.5 # pixel intensity constant for background removal

    resources:
    threads: 1
    workers: 2
    procs: 5
    memory: 30GB


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

    owl submit pipeline --conf imc_pipeline.yaml

The command above returns a ``jobID`` number.

**Check status in the command line**

Pipeline status and log messages can be checked using:

.. code-block:: python

    owl pipeline status jobID

where ``jobID`` is the pipeline number.

**Check status in the archive**

Progress can be monitored from the Web at https://imaxt.ast.cam.ac.uk/archive/owl/