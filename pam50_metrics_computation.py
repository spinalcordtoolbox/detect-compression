''' activate venv: 

source /Users/etiennedufayet/spinalcordtoolbox/python/etc/profile.d/conda.sh
conda activate venv_sct

'''
import subprocess
import os
import pprint
import pandas as pd
import csv
import nibabel as nib
import math
import numpy as np


def get_slices_in_PAM50(compressed_level_dict, df_metrics, df_metrics_PAM50):
    """
    Get corresponding slice of compression in PAM50 space.
    :param compressed_level_dict: dict: Dictionary of levels and corresponding slice(s).
    :param df_metrics: pandas.DataFrame: Metrics output of sct_process_segmentation.
    :param df_metrics_PAM50: pandas.DataFrame: Metrics output of sct_process_segmentation in PAM50 anatomical dimensions.
    :return compression_level_dict_PAM50:
    """
    compression_level_dict_PAM50 = {}
    # Drop empty columns
    df_metrics_PAM50 = df_metrics_PAM50.drop(columns=['SUM(length)', 'DistancePMJ'])
    # Drop empty rows so they are not included for interpolation
    df_metrics_PAM50 = df_metrics_PAM50.dropna(axis=0)
    # Loop across slices and levels with compression
    for i, info in compressed_level_dict.items():
        compression_level_dict_PAM50[i] = {}
        for level, slices in info.items():
            level = int(level)
            # Number of slices in native image
            nb_slices_level = len(df_metrics.loc[df_metrics['VertLevel'] == level, 'VertLevel'].to_list())
            # Number of slices in PAM50
            nb_slices_PAM50 = len(df_metrics_PAM50.loc[df_metrics_PAM50['VertLevel'] == level, 'VertLevel'].to_list())
            # Do interpolation from native space to PAM50
            x_PAM50 = np.arange(0, nb_slices_PAM50, 1)
            x = np.linspace(0, nb_slices_PAM50 - 1, nb_slices_level)
            new_slices_coord = np.interp(x_PAM50, x,
                                         df_metrics.loc[df_metrics['VertLevel'] == level, 'Slice (I->S)'].to_list())
            # find nearest index
            slices_PAM50 = np.array([])
            for slice in slices:
                # get index corresponding to the min value
                idx = np.argwhere((np.round(new_slices_coord) - slice) == 0).T[0]  # Round to get all slices within ±1 arround the slice
                new_slice = [df_metrics_PAM50.loc[df_metrics_PAM50['VertLevel'] == level, 'Slice (I->S)'].to_list()[id] for id in idx]
                slices_PAM50 = np.append(slices_PAM50, new_slice, axis=0)
            slices_PAM50 = slices_PAM50.tolist()
            slices_PAM50 = list(map(int, slices_PAM50))
            compression_level_dict_PAM50.setdefault(i, {})[level] = slices_PAM50
    return compression_level_dict_PAM50


def check_compressed(row):
    for key, inner_dict in dict_pam50.items():
        for sublist in inner_dict.values():
            if row['Slice (I->S)'] in sublist:
                return 1
    return 0


path_to_dataset = '/Users/etiennedufayet'
path_to_sct = '/Users/etiennedufayet'

#récupère tous les subdirs d'un fichier = ici le fichier des patients
def get_subdirs(root_dir):
    subdirs = []
    for entry in os.scandir(root_dir):
        if entry.is_dir() and entry.name.startswith("sub"): ##on ne garde que ceux commençant par "sub"
            subdir_name = os.path.basename(entry.path)
            subdirs.append(subdir_name)
            subdirs.extend(get_subdirs(entry.path))
    return subdirs   

patients = get_subdirs(path_to_dataset+"/dcm-zurich")


full_dataset_pam50 = pd.DataFrame()

for patient in patients: 

    input_seg = path_to_dataset+"/dcm-zurich/derivatives/labels/"+patient+"/anat/"+patient+"_acq-axial_T2w_label-SC_mask-manual.nii.gz"
    label_file = path_to_dataset+"/dcm-zurich/derivatives/labels/"+patient+"/anat/"+patient+"_acq-axial_T2w_labels-manual.nii.gz"
    compression_file = path_to_dataset+"/dcm-zurich/derivatives/labels/"+patient+"/anat/"+patient+"_acq-axial_T2w_label-compression-manual.nii.gz"

    output_file = '/Users/etiennedufayet/Desktop/STAGE_3A/Compression_detection_zurich/pam50_metrics/metrics_per_subject/'+patient+'.csv'
    output_file_is_compressed = '/Users/etiennedufayet/Desktop/STAGE_3A/Compression_detection_zurich/pam50_metrics/metrics_per_subject_pam50_is_compressed/'+patient+'.csv'
    output_file_pam50 = '/Users/etiennedufayet/Desktop/STAGE_3A/Compression_detection_zurich/pam50_metrics/metrics_per_subject_pam50/'+patient+'_pam50.csv'
    output_discfile = '/Users/etiennedufayet/Desktop/STAGE_3A/Compression_detection_zurich/pam50_metrics/discfile_per_subject/'+patient+'_labeled.nii.gz'


    if  os.path.exists(input_seg) and os.path.exists(label_file) and os.path.exists(compression_file):

        command = ['python', path_to_sct+'/spinalcordtoolbox/spinalcordtoolbox/scripts/sct_label_utils.py',  '-i', input_seg, '-disc', label_file, '-o', output_discfile]
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        output, error = process.communicate()

        if os.path.exists(output_discfile): 

            input_discfile = output_discfile

            compression_img = nib.load(compression_file)
            compression_data = compression_img.get_fdata()
                    
            label_img = nib.load(input_discfile)
            label_data = label_img.get_fdata()


            
            ### 1ere partie: on constitue le dataset avec les metriques dans l'original space, et surtout on compute les vertlevels pour chaque slice
            command = ['python', path_to_sct+'/spinalcordtoolbox/spinalcordtoolbox/scripts/sct_process_segmentation.py',  '-i', input_seg, '-vertfile', input_discfile, '-perslice', '1', '-v', '2', '-o', output_file]
            process = subprocess.Popen(command, stdout=subprocess.PIPE)
            output, error = process.communicate()


            ## prepare labels 
            command = ['python', path_to_sct+'/spinalcordtoolbox/spinalcordtoolbox/scripts/sct_label_utils.py','-i',input_discfile, '-display']
            process = subprocess.Popen(command, stdout=subprocess.PIPE)
            output, error = process.communicate()

            output = output.decode()
            output_lines = output.splitlines()
            all_labels = output_lines[-1]

            paires = all_labels.split(":")
            dict_xyz_dict = {}

            # Parcourir les paires clé-valeur
            for paire in paires:
                # Diviser la paire en clés et valeurs individuelles
                elements = paire.split(",")
                
                # On supprime les espaces
                elements = [element.strip() for element in elements]
                
                disc = elements[-1]
                xyz = elements[:-1]

                ## on a constitué le dictionnaire disque: (x,y,z) depuis les labels 

                dict_xyz_dict[disc] = xyz
            

            unsorted_compression_levels = {}

            for x in range(compression_data.shape[0]):
                for y in range(compression_data.shape[1]):
                    for z in range(compression_data.shape[2]):
                        if compression_data[x, y, z] == 1: 

                            z_already_tackled = False

                            # On cherche la clé du dictionnaire la plus proche de z
                            for cle in reversed(dict_xyz_dict.keys()):
                                disc_slice = int(dict_xyz_dict[cle][2])
                                
                                if z_already_tackled == False :

                                    if z <= disc_slice : 
                                        z_already_tackled = True

                                        if cle in unsorted_compression_levels:
                                            unsorted_compression_levels[cle].append(z)
                                        else:
                                            unsorted_compression_levels[cle] = [z]
            
            sorted_compression_levels = {key: unsorted_compression_levels[key] for key in sorted(unsorted_compression_levels)}

            indexed_compression_levels = {index: {key:value} for index, (key, value) in enumerate(sorted_compression_levels.items())}


            command = ['python', path_to_sct+'/spinalcordtoolbox/spinalcordtoolbox/scripts/sct_process_segmentation.py','-i',input_seg, '-vertfile', input_discfile,  '-perslice',  '1', '-normalize-PAM50', '1',  '-v', '2', '-o', output_file_pam50]
            process = subprocess.Popen(command, stdout=subprocess.PIPE)
            output, error = process.communicate()



            df_metrics = pd.read_csv(output_file)
            df_metrics_pam50 = pd.read_csv(output_file_pam50)

            dict_pam50 = get_slices_in_PAM50(indexed_compression_levels, df_metrics, df_metrics_pam50)

            df_metrics_pam50['is_compressed'] = df_metrics_pam50.apply(check_compressed, axis=1)

            df_metrics_pam50 = df_metrics_pam50.filter(regex=r'^(?!STD)')
            df_metrics_pam50 = df_metrics_pam50.drop('SUM(length)', axis = 1)
            df_metrics_pam50 = df_metrics_pam50.drop('DistancePMJ', axis = 1)
            df_metrics_pam50 = df_metrics_pam50.dropna(axis=0)
            df_metrics_pam50 = df_metrics_pam50.drop('Filename', axis = 1)
            df_metrics_pam50 = df_metrics_pam50.drop('SCT Version', axis = 1)
            df_metrics_pam50 = df_metrics_pam50.drop('Timestamp', axis = 1)
            df_metrics_pam50 = df_metrics_pam50.rename(columns=lambda x: x.replace('MEAN(', '').replace(')', '') if x.startswith('MEAN') else x)
            df_metrics_pam50 = df_metrics_pam50.rename(columns={'Slice (I->S)': 'slice'})
            df_metrics_pam50 = df_metrics_pam50.reset_index(drop=True)

            df_metrics_pam50['Torsion'] = ''

            for index in df_metrics_pam50.index:
                
                if (index - 3 >= 0) and (index + 3 <= np.max(df_metrics_pam50.index)):

                        if df_metrics_pam50['orientation'].iloc[index] is not None and \
                        df_metrics_pam50['orientation'].iloc[index-1] is not None and \
                        df_metrics_pam50['orientation'].iloc[index+1] is not None and \
                        df_metrics_pam50['orientation'].iloc[index-2] is not None and \
                        df_metrics_pam50['orientation'].iloc[index+2] is not None and \
                        df_metrics_pam50['orientation'].iloc[index-3] is not None and \
                        df_metrics_pam50['orientation'].iloc[index+3] is not None:
                        
                    
                    
                            df_metrics_pam50['Torsion'].iloc[index] = 1/6 * (abs(df_metrics_pam50['orientation'].iloc[index] -
                                                                            df_metrics_pam50['orientation'].iloc[index-1]) +
                                                                        abs(df_metrics_pam50['orientation'].iloc[index] -
                                                                            df_metrics_pam50['orientation'].iloc[index+1]) +
                                                                        abs(df_metrics_pam50['orientation'].iloc[index-1] -
                                                                            df_metrics_pam50['orientation'].iloc[index-2]) +
                                                                        abs(df_metrics_pam50['orientation'].iloc[index+1] -
                                                                            df_metrics_pam50['orientation'].iloc[index+2]) +
                                                                        abs(df_metrics_pam50['orientation'].iloc[index-2] -
                                                                            df_metrics_pam50['orientation'].iloc[index-3]) +
                                                                        abs(df_metrics_pam50['orientation'].iloc[index+2] -
                                                                            df_metrics_pam50['orientation'].iloc[index+3]))
                    
                                    

                                        


                elif (index - 2 == 0) or (index + 2 == np.max(df_metrics_pam50.index)): 

                        if df_metrics_pam50['orientation'].iloc[index] is not None and \
                        df_metrics_pam50['orientation'].iloc[index-1] is not None and \
                        df_metrics_pam50['orientation'].iloc[index+1] is not None and \
                        df_metrics_pam50['orientation'].iloc[index-2] is not None and \
                        df_metrics_pam50['orientation'].iloc[index+2] is not None:
                            
                            df_metrics_pam50['Torsion'].iloc[index] = 1/4 * (abs(df_metrics_pam50['orientation'].iloc[index] -
                                                                            df_metrics_pam50['orientation'].iloc[index-1]) +
                                                                        abs(df_metrics_pam50['orientation'].iloc[index] -
                                                                            df_metrics_pam50['orientation'].iloc[index+1]) +
                                                                        abs(df_metrics_pam50['orientation'].iloc[index-1] -
                                                                            df_metrics_pam50['orientation'].iloc[index-2]) +
                                                                        abs(df_metrics_pam50['orientation'].iloc[index+1] -
                                                                            df_metrics_pam50['orientation'].iloc[index+2]))
                        
                    

                    
                        

                elif (index - 1 == 0) or (index + 1 == np.max(df_metrics_pam50.index)): 

                    if df_metrics_pam50['orientation'].iloc[index] is not None and \
                    df_metrics_pam50['orientation'].iloc[index-1] is not None and \
                    df_metrics_pam50['orientation'].iloc[index+1] is not None :
                        
                        df_metrics_pam50['Torsion'].iloc[index] = 1/2 * (abs(df_metrics_pam50['orientation'].iloc[index] -
                                                                            df_metrics_pam50['orientation'].iloc[index-1]) +
                                                                        abs(df_metrics_pam50['orientation'].iloc[index] -
                                                                            df_metrics_pam50['orientation'].iloc[index+1]))
                        
                                                                
                    
                else:
                    df_metrics_pam50['Torsion'].iloc[index] = None



            full_dataset_pam50 =  pd.concat([full_dataset_pam50, df_metrics_pam50])

full_dataset_pam50.to_csv('/Users/etiennedufayet/Desktop/STAGE_3A/Compression_detection_zurich/pam50_metrics/full_dataset_pam50.csv')



