''' activate venv: 

source /Users/etiennedufayet/spinalcordtoolbox/python/etc/profile.d/conda.sh
conda activate venv_sct

'''

import subprocess
import os
import nibabel as nib



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


## compute a dict of subjects - compressed discs with Jan's algorithm

input_mri_list = []
input_seg_list = []
input_discfile_list = []

for patient in patients: 
    input_mri = "/Users/etiennedufayet/dcm-zurich/"+patient+"/anat/"+patient+"_acq-axial_T2w.nii.gz"
    input_seg = "/Users/etiennedufayet/dcm-zurich/derivatives/labels/"+patient+"/anat/"+patient+"_acq-axial_T2w_label-SC_mask-manual.nii.gz"
    input_discfile = "/Users/etiennedufayet/dcm-zurich/derivatives/labels/"+patient+"/anat/"+patient+"_acq-axial_T2w_labels-manual.nii.gz"
    
    if os.path.exists(input_mri) and os.path.exists(input_seg) and os.path.exists(input_discfile):
        input_mri_list.append(input_mri)
        input_seg_list.append(input_seg)
        input_discfile_list.append(input_discfile)
        

# create the dictionnary with patient-compression: 
patient_compression_dict = {}

# Loop over input files
for input_mri, input_seg, input_discfile, patient in zip(input_mri_list, input_seg_list, input_discfile_list, patients):
    # Run the Python code with the input file as an argument
    
    command = ['python', '/Users/etiennedufayet/spinalcordtoolbox/spinalcordtoolbox/scripts/V2_sct_detect_compression.py','-i',input_mri, '-s', input_seg,  '-discfile', input_discfile]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, error = process.communicate()

    # Save the output to a file
    #output_file = input_file + '.out'
    #with open(output_file, 'w') as f:
    #    f.write(output.decode())

    
    output = output.decode()
    output_lines = output.splitlines()
    if output_lines:
        compressed_discs = output_lines[-1]
        if compressed_discs == []: 
            patient_compression_dict[patient] = None
        
        else :
            patient_compression_dict[patient] = compressed_discs



    
## compute a dict of subjects - compressed disc thanks to manually labeled files  
manual_patient_compression_disc = {}

for patient in patients: 
    
    compression_file = "/Users/etiennedufayet/dcm-zurich/derivatives/labels/"+patient+"/anat/"+patient+"_acq-axial_T2w_label-compression-manual.nii.gz"
    label_file = "/Users/etiennedufayet/dcm-zurich/derivatives/labels/"+patient+"/anat/"+patient+"_acq-axial_T2w_labels-manual.nii.gz"

    ## prepare data
    if os.path.exists(compression_file) and os.path.exists(label_file):
        
        compression_img = nib.load(compression_file)
        compression_data = compression_img.get_fdata()
        label_img = nib.load(label_file)
        label_data = label_img.get_fdata()


        ## prepare labels 
        command = ['python', '/Users/etiennedufayet/spinalcordtoolbox/spinalcordtoolbox/scripts/sct_label_utils.py','-i',label_file, '-display']
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        output, error = process.communicate()

        output = output.decode()
        output_lines = output.splitlines()
        all_labels = output_lines[-1]

        # Diviser la chaîne all_labels à chaque ":"
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

        # on parcourt le fichier des compressions à la recherche de là où elle se trouve

        for x in range(compression_data.shape[0]):
            for y in range(compression_data.shape[1]):
                for z in range(compression_data.shape[2]):
                    if compression_data[x, y, z] == 1: 

                        # On cherche la clé du dictionnaire la plus proche de z

                        differences = {cle: abs(int(valeur[2]) - z) for cle, valeur in dict_xyz_dict.items()}
                        nearest_disc = min(differences, key=lambda k: differences[k])
                        
                        #on ajoute au dictionnaire des compressions manuelles le disque comprimé, avec en clé le numéro du patient 

                        if patient in manual_patient_compression_disc:
                            manual_patient_compression_disc[patient].append(nearest_disc)
                        else:
                            manual_patient_compression_disc[patient] = [nearest_disc]


## on supprime les doublons de chaque dictionnaire s'il y en a 

for cle in manual_patient_compression_disc:
    valeurs = manual_patient_compression_disc[cle]
    
    # Supprimer les doublons des valeurs associées à la clé
    valeurs_sans_doublons = list(set(valeurs))
    
    # Mettre à jour les valeurs associées à la clé dans le dictionnaire
    manual_patient_compression_disc[cle] = valeurs_sans_doublons


for cle in patient_compression_dict:
    valeurs = patient_compression_dict[cle]
    
    # Supprimer les doublons des valeurs associées à la clé
    valeurs_sans_doublons = list(set(valeurs))
    
    # Mettre à jour les valeurs associées à la clé dans le dictionnaire
    patient_compression_dict[cle] = valeurs_sans_doublons

## on compare les deux dictionnaires (manual et avec le processing )

nb_bonnes_detection = 0
nb_erreurs = 0
# Parcourir les clés communes aux deux dictionnaires
for cle in set(manual_patient_compression_disc.keys()).intersection(patient_compression_dict.keys()):
    valeurs_dict1 = set(manual_patient_compression_disc[cle])
    valeurs_dict2 = set(patient_compression_dict[cle])

    # Comparer les valeurs communes et non communes
    valeurs_communes = valeurs_dict1.intersection(valeurs_dict2)
    valeurs_non_communes_dict1 = valeurs_dict1 - valeurs_communes
    valeurs_non_communes_dict2 = valeurs_dict2 - valeurs_communes

    nb_bonnes_detection += len(valeurs_communes)
    nb_erreurs += max(len(valeurs_non_communes_dict1), len(valeurs_non_communes_dict2))


print("nb_bonnes_detection =" +str(nb_bonnes_detection))
print("nb_erreurs =" +str(nb_erreurs))



'''
#test of a specific file 

input_mri = input_mri_list[1]
input_seg = input_seg_list[1]
input_discfile = input_discfile_list[1]


command = ['python', '/Users/etiennedufayet/spinalcordtoolbox/spinalcordtoolbox/scripts/sct_detect_compression.py','-i',input_mri, '-s', input_seg,  '-discfile', input_discfile]
process = subprocess.Popen(command, stdout=subprocess.PIPE)
output, error = process.communicate()

# Save the output to a file
#output_file = input_file + '.out'
#with open(output_file, 'w') as f:
#    f.write(output.decode())

output = output.decode()
output_lines = output.splitlines()
nb_compression = output_lines[-1]
nb_compression = int(nb_compression[0])
last_lines = output_lines[-nb_compression-1:-1]

compressed_discs = []

for i in range(nb_compression): 
    compressed_discs.append(int(last_lines[i]))

print(compressed_discs)


print(nb_compression)
'''