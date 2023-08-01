#Automatic detection of spinal cord compression

This repository contains scripts to perform automatic compression detection.

Currently, we use the following datasets: dcm_zurich - DCM patients inspired - DCM patients spine-generic data-multi-subject, r20230223 - healthy subjects and mild compressions

For each subject, we do the following:

- segment the spinal cord with sct_deepseg_sc and label vertebral levels with sct_label_vertebrae
- manually label the compression levels
- compute morphometric measures in PAM50 space with sct_process_segmentation -normalize-PAM50 1
- bring compression labels to the PAM50 space

Then, classification of the compression (is compression 0/1) is performed with a XGBoost classifier or a neural network.






