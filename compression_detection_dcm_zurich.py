''' activate venv: 

source /Users/etiennedufayet/spinalcordtoolbox/python/etc/profile.d/conda.sh
conda activate venv_sct

'''

import os
import pprint
import pandas as pd
import csv


from spinalcordtoolbox.process_seg import compute_shape
from spinalcordtoolbox.utils.fs import get_absolute_path
from spinalcordtoolbox.centerline.core import ParamCenterline
from spinalcordtoolbox.aggregate_slicewise import aggregate_per_slice_or_level, merge_dict, func_wa, func_std





#récupère tous les subdirs d'un fichier = ici le fichier des patients
def get_subdirs(root_dir):
    subdirs = []
    for entry in os.scandir(root_dir):
        if entry.is_dir() and entry.name.startswith("sub"): ##on ne garde que ceux commençant par "sub"
            subdir_name = os.path.basename(entry.path)
            subdirs.append(subdir_name)
            subdirs.extend(get_subdirs(entry.path))
    return subdirs   

patients = get_subdirs("/Users/etiennedufayet/dcm-zurich")


patient_seg_dict = {}
group_funcs = (('MEAN', func_wa), ('STD', func_std))

for patient in patients: 
    input_seg = "/Users/etiennedufayet/dcm-zurich/derivatives/labels/"+patient+"/anat/"+patient+"_acq-axial_T2w_label-SC_mask-manual.nii.gz"
    input_discfile = "/Users/etiennedufayet/dcm-zurich/derivatives/labels/"+patient+"/anat/"+patient+"_acq-axial_T2w_labels-manual.nii.gz"


    if  os.path.exists(input_seg) and os.path.exists(input_discfile):

        ## args for metrics computation
        fname_seg = input_seg
        fname_disc = input_discfile
        angle_correction = 1
        param_centerline = ParamCenterline(
                algo_fitting='bspline',
                smooth=30,
                minmax=True)
        verbose = 1
        torsion_slices = 3

       # Compute morphometric metrics
        metrics, fit_results = compute_shape(fname_seg,
                                         angle_correction=angle_correction,
                                         param_centerline=param_centerline,
                                         verbose=verbose)
        
        print(metrics)
        
        # Compute the average and standard deviation across slices
        metrics_agg = {}
        for key in ['area', 'diameter_AP', 'diameter_RL', 'solidity', 'orientation']:
            # Note: we do not need to calculate all the metrics, we need just:
            #   - area (will be CSA)
            #   - diameter_AP and diameter_RL (used to calculate compression ratio)
            #   - solidity
            #   - orientation (used to calculate torsion)
            # Note: we have to calculate metrics across all slices (perslice) to be able to compute orientation
            metrics_agg[key] = aggregate_per_slice_or_level(metrics[key],
                                                            perslice=True,
                                                            perlevel=False, fname_vert_level=fname_disc,
                                                            group_funcs=group_funcs
                                                            )

        metrics_agg_merged = merge_dict(metrics_agg)

        # Compute compression ratio (CR) as 'diameter_AP' / 'diameter_RL'
        # TODO - compression ratio (CR) could be computed directly within the compute_shape function -> consider that
        for key in metrics_agg_merged.keys():           # Loop across slices
            # Ignore slices which have diameter_AP or diameter_RL equal to None (e.g., due to bad SC segmentation)
            if metrics_agg_merged[key]['MEAN(diameter_AP)'] is None or metrics_agg_merged[key]['MEAN(diameter_RL)'] is None:
                metrics_agg_merged[key]['CompressionRatio'] = None
            else:
                metrics_agg_merged[key]['CompressionRatio'] = metrics_agg_merged[key]['MEAN(diameter_AP)'] / \
                                                            metrics_agg_merged[key]['MEAN(diameter_RL)']

        # Compute torsion as the average of absolute differences in orientation between the given slice and x slice(s)
        # above and below. For details see eq 1-3 in https://pubmed.ncbi.nlm.nih.gov/35371944/
        # TODO - torsion could be computed directly within the compute_shape function -> consider that
        # Since the torsion is computed from slices above and below, it cannot be computed for the x first and last x slices
        # --> x first and x last slices will be excluded f
        # From the torsion computation
        # For example, if torsion_slices == 3, the first three and last three slices will have torsion = None
        slices = list(metrics_agg_merged.keys())[torsion_slices:-torsion_slices]

        for key in metrics_agg_merged.keys():  # Loop across slices
            if key in slices:
                # Note: the key is a tuple (e.g. `1,`), not an int (e.g., 1), thus key[0] is used to convert tuple to int
                # and `,` is used to convert int back to tuple
                # TODO - the keys could be changed from tuple to int inside the compute_shape function -> consider that
                if metrics_agg_merged[key]['MEAN(orientation)'] is not None and \
                metrics_agg_merged[key[0] - 1,]['MEAN(orientation)'] is not None and \
                metrics_agg_merged[key[0] + 1,]['MEAN(orientation)'] is not None and \
                metrics_agg_merged[key[0] - 2,]['MEAN(orientation)'] is not None and \
                metrics_agg_merged[key[0] + 2,]['MEAN(orientation)'] is not None and \
                metrics_agg_merged[key[0] - 3,]['MEAN(orientation)'] is not None and \
                metrics_agg_merged[key[0] + 3,]['MEAN(orientation)'] is not None:
                
                
                
                    if torsion_slices == 3:
                        metrics_agg_merged[key]['Torsion'] = 1/6 * (abs(metrics_agg_merged[key]['MEAN(orientation)'] -
                                                                        metrics_agg_merged[key[0] - 1,]['MEAN(orientation)']) +
                                                                    abs(metrics_agg_merged[key]['MEAN(orientation)'] -
                                                                        metrics_agg_merged[key[0] + 1,]['MEAN(orientation)']) +
                                                                    abs(metrics_agg_merged[key[0] - 1,]['MEAN(orientation)'] -
                                                                        metrics_agg_merged[key[0] - 2,]['MEAN(orientation)']) +
                                                                    abs(metrics_agg_merged[key[0] + 1,]['MEAN(orientation)'] -
                                                                        metrics_agg_merged[key[0] + 2,]['MEAN(orientation)']) +
                                                                    abs(metrics_agg_merged[key[0] - 2,]['MEAN(orientation)'] -
                                                                        metrics_agg_merged[key[0] - 3,]['MEAN(orientation)']) +
                                                                    abs(metrics_agg_merged[key[0] + 2,]['MEAN(orientation)'] -
                                                                        metrics_agg_merged[key[0] + 3,]['MEAN(orientation)']))
                        # TODO - implement also equations for torsion_slices == 1 and torsion_slices == 2
            else:
                metrics_agg_merged[key]['Torsion'] = None

    patient_seg_dict[patient] = metrics_agg_merged





## Test for only one patient 
'''patient = patients[0] 
input_seg = "/Users/etiennedufayet/dcm-zurich/derivatives/labels/"+patient+"/anat/"+patient+"_acq-axial_T2w_label-SC_mask-manual.nii.gz"
input_discfile = "/Users/etiennedufayet/dcm-zurich/derivatives/labels/"+patient+"/anat/"+patient+"_acq-axial_T2w_labels-manual.nii.gz"

if  os.path.exists(input_seg) and os.path.exists(input_discfile):

    ## args for metrics computation
    fname_seg = input_seg
    fname_disc = input_discfile
    angle_correction = 1
    param_centerline = ParamCenterline(
            algo_fitting='bspline',
            smooth=30,
            minmax=True)
    verbose = 1
    torsion_slices = 3

    # Compute morphometric metrics
    metrics, fit_results = compute_shape(fname_seg,
                                        angle_correction=angle_correction,
                                        param_centerline=param_centerline,
                                        verbose=verbose)
        
    # Compute the average and standard deviation across slices
    metrics_agg = {}
    for key in ['area', 'diameter_AP', 'diameter_RL', 'solidity', 'orientation']:
        # Note: we do not need to calculate all the metrics, we need just:
        #   - area (will be CSA)
        #   - diameter_AP and diameter_RL (used to calculate compression ratio)
        #   - solidity
        #   - orientation (used to calculate torsion)
        # Note: we have to calculate metrics across all slices (perslice) to be able to compute orientation
        metrics_agg[key] = aggregate_per_slice_or_level(metrics[key],
                                                        perslice=True,
                                                        perlevel=False, fname_vert_level=fname_disc,
                                                        group_funcs=group_funcs
                                                        )

    metrics_agg_merged = merge_dict(metrics_agg)

    # Compute compression ratio (CR) as 'diameter_AP' / 'diameter_RL'
    # TODO - compression ratio (CR) could be computed directly within the compute_shape function -> consider that
    for key in metrics_agg_merged.keys():           # Loop across slices
        # Ignore slices which have diameter_AP or diameter_RL equal to None (e.g., due to bad SC segmentation)
        if metrics_agg_merged[key]['MEAN(diameter_AP)'] is None or metrics_agg_merged[key]['MEAN(diameter_RL)'] is None:
            metrics_agg_merged[key]['CompressionRatio'] = None
        else:
            metrics_agg_merged[key]['CompressionRatio'] = metrics_agg_merged[key]['MEAN(diameter_AP)'] / \
                                                        metrics_agg_merged[key]['MEAN(diameter_RL)']

    # Compute torsion as the average of absolute differences in orientation between the given slice and x slice(s)
    # above and below. For details see eq 1-3 in https://pubmed.ncbi.nlm.nih.gov/35371944/
    # TODO - torsion could be computed directly within the compute_shape function -> consider that
    # Since the torsion is computed from slices above and below, it cannot be computed for the x first and last x slices
    # --> x first and x last slices will be excluded f
    # From the torsion computation
    # For example, if torsion_slices == 3, the first three and last three slices will have torsion = None
    slices = list(metrics_agg_merged.keys())[torsion_slices:-torsion_slices]

    for key in metrics_agg_merged.keys():  # Loop across slices
        if key in slices:
            # Note: the key is a tuple (e.g. `1,`), not an int (e.g., 1), thus key[0] is used to convert tuple to int
            # and `,` is used to convert int back to tuple
            # TODO - the keys could be changed from tuple to int inside the compute_shape function -> consider that
            if metrics_agg_merged[key]['MEAN(orientation)'] is not None and \
            metrics_agg_merged[key[0] - 1,]['MEAN(orientation)'] is not None and \
            metrics_agg_merged[key[0] + 1,]['MEAN(orientation)'] is not None and \
            metrics_agg_merged[key[0] - 2,]['MEAN(orientation)'] is not None and \
            metrics_agg_merged[key[0] + 2,]['MEAN(orientation)'] is not None and \
            metrics_agg_merged[key[0] - 3,]['MEAN(orientation)'] is not None and \
            metrics_agg_merged[key[0] + 3,]['MEAN(orientation)'] is not None:
            
            
            
                if torsion_slices == 3:
                    metrics_agg_merged[key]['Torsion'] = 1/6 * (abs(metrics_agg_merged[key]['MEAN(orientation)'] -
                                                                    metrics_agg_merged[key[0] - 1,]['MEAN(orientation)']) +
                                                                abs(metrics_agg_merged[key]['MEAN(orientation)'] -
                                                                    metrics_agg_merged[key[0] + 1,]['MEAN(orientation)']) +
                                                                abs(metrics_agg_merged[key[0] - 1,]['MEAN(orientation)'] -
                                                                    metrics_agg_merged[key[0] - 2,]['MEAN(orientation)']) +
                                                                abs(metrics_agg_merged[key[0] + 1,]['MEAN(orientation)'] -
                                                                    metrics_agg_merged[key[0] + 2,]['MEAN(orientation)']) +
                                                                abs(metrics_agg_merged[key[0] - 2,]['MEAN(orientation)'] -
                                                                    metrics_agg_merged[key[0] - 3,]['MEAN(orientation)']) +
                                                                abs(metrics_agg_merged[key[0] + 2,]['MEAN(orientation)'] -
                                                                    metrics_agg_merged[key[0] + 3,]['MEAN(orientation)']))
                    # TODO - implement also equations for torsion_slices == 1 and torsion_slices == 2
        else:
            metrics_agg_merged[key]['Torsion'] = None

patient_seg_dict[patient] = metrics_agg_merged

pprint.pprint(patient_seg_dict, indent=4)'''
