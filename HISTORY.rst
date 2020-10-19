=======
History
=======

0.10.0
------

* Add support for Zarr files.
* Remove support for TIFF files.
* Support for OpenCV 4.4.
* Automatic normalisation factor.
* Behaviour changed back so all RoIs are now processed
  in one run.

0.3.2 (2020-06-01)
------------------

* Bug fixes.
* Raise error when pipeline fails.
* Add an extra keyword to run the processing in draft mode.
* Add support for big tiff files.
* Add support for non ome-tif files with a directory structure
  and file naming compatible with histocat.
* Improve validation of configuration keywords.
* The location of input data changes from `img_path` to
  `input_path`.
* IMC slides have now to be processed independently (i.e.
  `input_path` is the location of one IMC slide.

0.2.0 (2020-01-20)
------------------

* Add contribution guide to documentation.
* Update docstrings and documents.
* Add a keyword that marks intensity correction optional
  (aic_apply_intensity_correction). by default it is False.
* Add the standard deviation of the Guassian kernel used
  to convolve the image to do intensity correction (aic_sigma).
* Output an image with detected objects overlayed on top of reference image.
* Move normalize_factor keyword to segmentation group.

0.1.7 (2019-10-08)
------------------

* Fix issue with tiff files out of order.

0.1.6 (2019-08-08)
------------------

* Updates to configuration file.
* Documentation updates.

0.1.5 (2019-07-23)
------------------

* First release.
