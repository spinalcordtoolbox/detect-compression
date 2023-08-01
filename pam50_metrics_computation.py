''' activate venv for SCT : 

source /Users/etiennedufayet/spinalcordtoolbox/python/etc/profile.d/conda.sh
conda activate venv_sct

'''
import subprocess
import os
import pandas as pd
import nibabel as nib
import numpy as np


'''
First of all, activate the sct environment displayed at the top of the script. 

The purpose of this python script is to build a dataset of PAM50 standardized metrics for classification purposes. 
PAM50 normalization is important for mixing data from different datasets. 

This script loops over each patient in the dataset to compute the metrics associated with each patient. 
The first step is to retrieve the names of each patient. This can be done in two different ways: 
- either from a text file containing patient names (for example, to generate a list of patients to be studied at random) -- use read_text_file function
- or from the dataset directory containing all patient files -- use get_subdirs function
These two functions build a list of patient names (strings), on which we can build the loop to obtain the metrics associated with each dataset slice. 

This script proceeds in 4 steps: to obtain the metrics and compression labels in PAM50, we need to: 
- Compute the metrics for each slice in the native space 
- retrieve the compression labels associated with each slice, and add them for each slice in the native space
- Convert metrics to PAM50
- Use a linear interpolation process to convert a compression label from native space to PAM50. 

It's the conversion of the compression label from native space to PAM50 space that makes the operation tricky. 

At each of these stages, methods are called, and input and output file names are requested to save the intermediate results. 

'''


'''
First, several functions are created: 
'''


'''
This function performs linear interpolation to convert compression labels in PAM50 space. 
It takes as input the metrics dataframe in native space, the compression label dictionary in native space, and the metrics dataframe in PAM50 space. 

All these inputs are constructed in the rest of the script. 
'''
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
            level = int(float(level))
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


'''
This function converts data from the PAM50 interpolated compression dictionary (after using get_slices_in_PAM50) into the PAM50 metrics dataframe. 
'''
def check_compressed(row):
    for key, inner_dict in dict_pam50.items():
        for sublist in inner_dict.values():
            if row['Slice (I->S)'] in sublist:
                return 1
    return 0


'''
This function creates the list of patient names mentioned in the introduction. This list is used to loop through the patient names. It is created directly from the dataset. 
'''
def get_subdirs(root_dir):
    subdirs = []
    for entry in os.scandir(root_dir):
        if entry.is_dir() and entry.name.startswith("sub"): ##on ne garde que ceux commençant par "sub"
            subdir_name = os.path.basename(entry.path)
            subdirs.append(subdir_name)
            subdirs.extend(get_subdirs(entry.path))
    return subdirs   


'''
This function creates the list of patient names mentioned in the introduction. This list is used to loop through the patient names. 
It is created from a text file containing patient names.  
'''
def read_texte_file(nom_fichier):
    liste_strings = []

    try:
        with open(nom_fichier, 'r') as fichier:
            for ligne in fichier:
                # Retirez le saut de ligne (\n) à la fin de chaque ligne
                ligne = ligne.strip()
                liste_strings.append(ligne)
        
        return liste_strings
    except IOError as e:
        print("Erreur lors de la lecture du fichier :", e)
        return None



## provide path to the script
path_to_dataset = '...'
path_to_sct = '...'


## Get the list of patient names and create a list with it :

# By the dataset and get_subdirs function: 
#patients = get_subdirs(path_to_dataset+"/dcm-zurich")
#patients = get_subdirs(path_to_dataset+"/inspired")
#patients = get_subdirs(path_to_dataset+"/data-multi-subject")


#By a text file
file_name = '...'
patients  = read_texte_file(file_name)


## initialize the dataframe of metrics 
full_dataset_pam50 = pd.DataFrame()


## start the loop over patients from the patient list 
for patient in patients: 

    ## provide input files and output files to compute the metrics 
    ## output files are used to store intermediate results 

    input_seg = path_to_dataset+"/inspired/derivatives/labels/"+patient+"/anat/"+patient+"_T2w_seg-manual.nii.gz"
    label_file = path_to_dataset+"/inspired/derivatives/labels/"+patient+"/anat/"+patient+"_T2w_labels-disc-manual.nii.gz"

    ## the compression file have to be used only for patho subjects, otherwise you can
    compression_file = path_to_dataset+"/inspired/derivatives/labels/"+patient+"/anat/"+patient+"_acq-axial_T2w_label-compression-manual.nii.gz"
    
    ## output files 
    output_file = '.../pam50_metrics/inspired/metrics_per_subject/'+patient+'.csv'
    output_file_is_compressed = '.../pam50_metrics/inspired/metrics_per_subject_pam50_is_compressed/'+patient+'.csv'
    output_file_pam50 = '.../pam50_metrics/inspired/metrics_per_subject_pam50/'+patient+'_pam50.csv'
    output_discfile = '.../pam50_metrics/inspired/discfile_per_subject/'+patient+'_labeled.nii.gz'


    if  os.path.exists(input_seg) and os.path.exists(label_file) and os.path.exists(compression_file):
    
        ## compute discfile and store it 
        command = ['python', path_to_sct+'/spinalcordtoolbox/spinalcordtoolbox/scripts/sct_label_utils.py',  '-i', input_seg, '-disc', label_file, '-o', output_discfile]
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        output, error = process.communicate()
        
    
        input_discfile = output_discfile

        ## load the compression file and get it as data 
        compression_img = nib.load(compression_file)
        compression_data = compression_img.get_fdata()
        
        
        ## load the disc file and get it as data 
        label_img = nib.load(input_discfile)
        label_data = label_img.get_fdata()

        '''
        We compute metrics in native space. 
        It's important to have computed discfile beforehand, as it's a necessary argument to sct_process_segmentation. 
        Here, sct_process_segmentation returns the metrics associated with segmentation in native space. 
        '''
        command = ['python', path_to_sct+'/spinalcordtoolbox/spinalcordtoolbox/scripts/sct_process_segmentation.py',  '-i', input_seg, '-vertfile', input_discfile, '-perslice', '1', '-v', '2', '-o', output_file]
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        output, error = process.communicate()
        

        ## We get labels from discfiles 
        command = ['python', path_to_sct+'/spinalcordtoolbox/spinalcordtoolbox/scripts/sct_label_utils.py','-i',input_discfile, '-display']
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        output, error = process.communicate()

        ## We transform the output to be able to use it 
        output = output.decode()
        output_lines = output.splitlines()
        all_labels = output_lines[-1]

        ## 
        paires = all_labels.split(":")
       
       
        '''
        The idea is to know the spatial extent of a disc over the course of slices. 
        This dictionary (dict_xyz_dict) carries the information: such-and-such a disk extends from such-and-such a slice to such-and-such a slice. 
        IMPORTANT: For a healthy subject, there is no compression, so do not use this section. 
        '''

        dict_xyz_dict = {}
        # Go through key-values paires 
        for paire in paires:
            # Divide the pair into individual keys and values
            elements = paire.split(",")
            
            # We delete spaces
            elements = [element.strip() for element in elements]
            
            disc = elements[-1]
            xyz = elements[:-1]

            ## we have created the disk dictionary: (x,y,z) from the labels

            dict_xyz_dict[disc] = xyz
        

        '''
        We loop over the compression file. As soon as a compression label is found (with its coordinates (x,y,z)),
        we compare it with dict_xyz to find out which disk is involved. z represents the slice.
        The result is a dictionary containing information on the compressions: each compressed slice and its disk. 
        It's important to have information on the disk, as this enables linear interpolation for subsequent conversion from compression label to PAM50. 

        As the compression labels can be found out of order in relation to the order of the slice numbers, 
        we first create an out-of-order dictionary with all the compressions, which we then sort.

        IMPORTANT: For a healthy subject, there is no compression, so do not use this section. 
        '''


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
        
        ## We compute metrics in the PAM50 space
        command = ['python', path_to_sct+'/spinalcordtoolbox/spinalcordtoolbox/scripts/sct_process_segmentation.py','-i',input_seg, '-vertfile', input_discfile,  '-perslice',  '1', '-normalize-PAM50', '1',  '-v', '2', '-o', output_file_pam50]
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        output, error = process.communicate()
        

        ## we get dataframes of metrics in the native and PAM50 spaceds
        df_metrics = pd.read_csv(output_file)
        df_metrics_pam50 = pd.read_csv(output_file_pam50)

        ## Linear interpolation is used to convert compression labels from native space to PAM 50 space.
        ## As an output, we get a dict of compressions in PAM50, that needs to be turned into dataframe feature
        dict_pam50 = get_slices_in_PAM50(indexed_compression_levels, df_metrics, df_metrics_pam50)

        ## Convert the dict of compressions in PAM50 to a feature in the dataframe
        df_metrics_pam50['is_compressed'] = df_metrics_pam50.apply(check_compressed, axis=1)

        ## For a healthy subject, all labels are set to 0
        #df_metrics_pam50['is_compressed'] = 0

        ## clean the dataframe
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


        ## compute torsion metric from orientation metric
        ## We use 3 slices under and above if available, otherwise 2, otherwise 1
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


        ## We concat the metrics dataframe computed from this patient with full_dataset_pam50
        full_dataset_pam50 =  pd.concat([full_dataset_pam50, df_metrics_pam50])

## export the dataframe to CSV file
full_dataset_pam50.to_csv('/Users/etiennedufayet/Desktop/STAGE_3A/Compression_detection_zurich/pam50_metrics/illness_detection/healthy_subjects/healthy_subjects_test/patients_multi_sub_train.csv')



