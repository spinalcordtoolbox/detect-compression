# Morphometric parameters- based compression detection with dcm_zurich dataset

commit number of the dataset (git_annex neuropoly): 3b2a27f8ab97ca5ca3409a4b7b1dbdc20e130dd3

The aim of my work in this repo is to perform a compression detection method on dcm-zurich dataset.
The approach will be ML-based, same as the one of sct_detect_compression.

[sct_detect_compression](https://github.com/spinalcordtoolbox/detect-compression/blob/main/sct_detect_compression.py) uses a probability formula that gives, for a given slice, its probability to be compressed:

<img width="703" alt="formule_proba" src="https://github.com/spinalcordtoolbox/detect-compression/assets/116156522/78e66291-924b-4891-8abf-a401b7f9f5e1">

For each slice of the spinal cord MRI images, we compute morphometric parameters (with sct_process_segmentation) that are arguments of this probability formula.
The coefficients of this formula have been determined on a [specific dataset](https://pubmed.ncbi.nlm.nih.gov/35371944/).

First, the idea is to figure out clearly whether or not we can directly apply the probability formula for other datasets (here dcm_zurich).

Then, the approach is to perform a regression on morphometric parameters of dcm_zurich dataset slices in order to find proper coefficients for the probability formula.






[loop_compression_detection](https://github.com/spinalcordtoolbox/detect-compression/blob/main/loop_compression_detection.py) performs sct_detect_compression on every subject of dcm_zurich and compares the output with the manual labelling.
Computation results highlight bad efficiency for this algorithm without proper coefficients in the probability formula.

[project_detect_compression](https://github.com/spinalcordtoolbox/detect-compression/blob/main/compression_detection_dcm_zurich.py) aims at finding proper coefficients for the probability formula, fitted with dcm_zurich dataset.

To do so, first, it does create a dataset organized as described below:

We represent each subject of the dataset whose manual compression labelling, spinal cord segmentation, and MRI images are available;
For each subject, we study every slice;
For each slice, a list of morphometric parameters obtained through the processing of sct_process_segmentation.
For each slice, a label: 1 if the slice is compressed, 0 otherwise.
Then, the idea is to perform stepwise linear or logistic regression on this dataset, and determine coefficients of the probability formula.







