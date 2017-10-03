import cPickle as pickle
import copy
import os

import menpo.io as mio
import numpy as np
from menpo.feature import no_op
from menpo.landmark import labeller, face_ibug_68_to_face_ibug_66_trimesh
from menpo.math import as_matrix
from menpo.model import PCAModel
from menpofit.aam import HolisticAAM, LucasKanadeAAMFitter
from menpofit.builder import (build_reference_frame, warp_images, align_shapes, rescale_images_to_reference_shape)
from menpofit.transform import DifferentiableAlignmentSimilarity
from pathlib import Path


#=======================================================================================================================
### Function for Reading {LFPW-AFW-Helen-Ibug} Images ###
#=======================================================================================================================
def load_image(i):
    i = i.crop_to_landmarks_proportion(0.5)
    if i.n_channels == 3:
        i = i.as_greyscale()

    labeller(i, 'PTS', face_ibug_68_to_face_ibug_66_trimesh)
    del i.landmarks['PTS']
    return i

#=======================================================================================================================
### Function for Reading Extended-Cohn-Kanade Images ###
#=======================================================================================================================
def Read_CK(land_tmp,image_path):
    img = mio.import_image(image_path)
    land_path = image_path[:-4] + "_landmarks.txt"
    with open(land_path) as file:
        tmp = np.array([[float(x) for x in line.split()] for line in file])

    # Removing Extra Points - Lips inner corner
    tmp = np.delete(tmp, (60), axis=0)
    tmp = np.delete(tmp, (63), axis=0)

    # Swapping Columns (Y,X) -> (X,Y)
    result = np.zeros((66,2))
    result [:,0] = tmp[:,1]
    result [:,1] = tmp[:,0]

    # Adding Landmarks
    land_tmp.lms.points= result
    img.landmarks['face_ibug_66_trimesh'] = land_tmp

    # Gray_Scaling
    img = img.crop_to_landmarks_proportion(0.5)
    if img.n_channels == 3:
        img = img.as_greyscale()

    return img

#=======================================================================================================================
### Function for Reading UNBC-McMaster Images ###
#=======================================================================================================================
def Read_UNBC(land_tmp,image_path):
    img = mio.import_image(image_path)
    land_path = image_path[:-4] + "_landmarks.txt"
    with open(land_path) as file:
        tmp = np.array([[float(x) for x in line.split()] for line in file])

    # Swapping Columns (Y,X) -> (X,Y)
    result = np.zeros((66,2))
    result [:,0] = tmp[:,1]
    result [:,1] = tmp[:,0]

    # Adding Landmarks
    land_tmp.lms.points= result
    img.landmarks['face_ibug_66_trimesh'] = land_tmp

    # Gray_Scaling
    img = img.crop_to_landmarks_proportion(0.5)
    if img.n_channels == 3:
        img = img.as_greyscale()

    return img

#=======================================================================================================================
### Loading Data ###
#=======================================================================================================================
# Loading the face_ibug_66_trimesh template
with open('/u/azinasg/Code/face_ibug_66_trimesh_temp.pkl', 'rb') as input:
    land_tmp = pickle.load(input)

# Loading (LFPW-AFW-Helen-Ibug) Images (Source Images)
source_path = Path('/u/azinasg/Research/Source_Small')
source_images = [load_image(i) for i in mio.import_images(source_path, verbose=True)]

# Loading CK Images (Source Images)
CK_root = "/u/azinasg/Research/Sample_CK+_Small"
for root, dirs, filenames in os.walk(CK_root):
    for filename in filenames:
        if (".png" in filename) and (".DS_Store" not in filename) :
            tmp_image = Read_CK(land_tmp,root+"/"+filename)
            source_images.append(tmp_image)

# Loading UNBC Images - Target
UNBC_root = "/u/azinasg/Research/Sample_UNBC_Small_Target"
target_images = []
for root, dirs, filenames in os.walk(UNBC_root):
    for filename in filenames:
        if (".png" in filename) and (".DS_Store" not in filename) :
            target_images.append(Read_UNBC(land_tmp,root+"/"+filename))

# Loading UNBC Images - Test
UNBC_root = "/u/azinasg/Research/Sample_UNBC_Small_Test_2"
test_images = []
for root, dirs, filenames in os.walk(UNBC_root):
    for filename in filenames:
        if (".png" in filename) and (".DS_Store" not in filename):
            test_images.append(Read_UNBC(land_tmp, root + "/" + filename))

#=======================================================================================================================
### Data Split Into Test, Target ###
#=======================================================================================================================
all_images = copy.deepcopy(target_images+source_images)

#=======================================================================================================================
### Pre-Computation ###
#=======================================================================================================================
# Building Source_AAM
source_aam = HolisticAAM(
    source_images,
    group='face_ibug_66_trimesh',
    holistic_features=no_op,
    scales=1,
    diagonal=150,
    max_appearance_components=200,
    max_shape_components=100,
    verbose=True
)

# Building Target AAM
target_aam = HolisticAAM(
    target_images,
    group='face_ibug_66_trimesh',
    holistic_features=no_op,
    scales=1,
    diagonal=150,
    max_appearance_components=200,
    max_shape_components=100,
    verbose=True
)

# Building SUT AAM
SUT_aam = HolisticAAM(
    all_images,
    group='face_ibug_66_trimesh',
    holistic_features=no_op,
    scales=1,
    diagonal=150,
    max_appearance_components=200,
    max_shape_components=100,
    verbose=True
)

#=======================================================================================================================
### Main Body ###
#=======================================================================================================================
for x in range(0,11,1):

    #===================================================================================================================
    ### Setting Hyper-Parameters ###
    #===================================================================================================================
    alpha = x/10.0
    beta = 0.5

    #===================================================================================================================
    ### Rescaling Images to the reference shape of SUT Model (Mean Shape of Model with Diagonal = 150) ###
    #===================================================================================================================
    ST = rescale_images_to_reference_shape(target_images, 'face_ibug_66_trimesh' ,
                                           SUT_aam.reference_shape, verbose=True)
    SS = rescale_images_to_reference_shape(source_images, 'face_ibug_66_trimesh' ,
                                           SUT_aam.reference_shape, verbose=True)

    #===================================================================================================================
    ### Building Shape Model ###
    #===================================================================================================================
    # Mean-Centering Target Sample Shapes
    ST_scaled_shapes = [i.landmarks['face_ibug_66_trimesh'].lms for i in ST]
    ST_aligned_shapes = align_shapes(ST_scaled_shapes)
    ST_data, ST_template = as_matrix(ST_aligned_shapes, return_template=True, verbose=True)
    ST_N = ST_data.shape[0]
    ST_mean = np.mean(ST_data, axis=0)
    ST_data -= ST_mean

    # Mean-Centering Source Sample Shapes
    SS_scaled_shapes = [i.landmarks['face_ibug_66_trimesh'].lms for i in SS]
    SS_aligned_shapes = align_shapes(SS_scaled_shapes)
    SS_data, SS_template = as_matrix(SS_aligned_shapes, return_template=True, verbose=True)
    SS_N = SS_data.shape[0]
    SS_mean = np.mean(SS_data, axis=0)
    SS_data -= SS_mean

    # Defining Target Weight Vector
    ST_weights = np.sqrt((alpha/float(ST_N))*np.ones(ST_N))
    WST = np.diag(ST_weights)
    ST_data = WST.dot(ST_data)

    # Defining Source Weight Vector
    Source_weights = np.ones(SS_N)
    SS_weights = np.sqrt(((1-alpha)/float(SS_N))*Source_weights)
    WSS = np.diag(SS_weights)
    SS_data = WSS.dot(SS_data)

    # Buidling Data for PCA
    S_Star = np.vstack((ST_data,SS_data))
    n_new_samples = ST_N + SS_N
    Shape_e_mean = ( beta*ST_N*ST_mean + (1-beta)*SS_N*SS_mean ) / (beta*ST_N + (1-beta)*SS_N)
    S_Star += Shape_e_mean

    # Computing PCA Model
    Shape_tmp = PCAModel(S_Star,centre=True,n_samples=n_new_samples,max_n_components=SUT_aam.max_shape_components[0]
                   ,inplace=True,verbose=True,azin_run=False,azin_temp=ST_template)

    # Setting Models info
    SUT_aam.shape_models[0].model=Shape_tmp

    SUT_aam.shape_models[0]._target = None

    SUT_aam.shape_models[0]._weights = np.zeros(SUT_aam.shape_models[0].model.n_active_components)

    SUT_aam.shape_models[0]._target = SUT_aam.shape_models[0].model.mean()

    mean = SUT_aam.shape_models[0].model.mean()

    SUT_aam.shape_models[0].global_transform = DifferentiableAlignmentSimilarity(mean, mean)
    # Re-orthonormalize
    SUT_aam.shape_models[0]._construct_similarity_model()
    # Set the target to the new mean
    SUT_aam.shape_models[0]._sync_target_from_state()

    #===================================================================================================================
    #### Shape Model Finished ###
    #===================================================================================================================
    ### Building Appearance Model ###
    #===================================================================================================================
    # Building SUT reference frame
    SUT_reference_frame = build_reference_frame(SUT_aam.reference_shape)

    # Obtain warped target samples
    ST_warped = warp_images(ST, ST_scaled_shapes, SUT_reference_frame, SUT_aam.transform,verbose=True)

    # Obtain warped source samples
    SS_warped = warp_images(SS, SS_scaled_shapes, SUT_reference_frame, SUT_aam.transform,verbose=True)

    # Building Data Matrix
    ST_App_data, ST_App_template = as_matrix(ST_warped, return_template=True, verbose=True)
    ST_App_N = ST_App_data.shape[0]
    SS_App_data, SS_App_template = as_matrix(SS_warped, return_template=True, verbose=True)
    SS_App_N = SS_App_data.shape[0]

    # Mean Centering the data
    SS_App_mean = np.mean(SS_App_data, axis=0)
    ST_App_mean = np.mean(ST_App_data, axis=0)
    SS_App_data -= SS_App_mean
    ST_App_data -= ST_App_mean

    # Building S* and Appling PCA on it
    SS_App_data = WSS.dot(SS_App_data)
    ST_App_data = WST.dot(ST_App_data)
    S_Star_App = np.vstack((ST_App_data,SS_App_data))
    App_n_new_samples = ST_App_N + SS_App_N

    # Calculating the Mean Appearance
    App_e_mean = ( beta*ST_App_N*ST_App_mean + (1-beta)*SS_App_N*SS_App_mean ) / ( beta*ST_App_N + (1-beta)*SS_App_N )
    S_Star_App += App_e_mean

    App_tmp = PCAModel(S_Star_App, centre=True, n_samples=n_new_samples,
                       max_n_components=SUT_aam.max_appearance_components[0],
                       inplace=True, verbose=True, azin_run=False, azin_temp=SS_App_template)

    del SUT_aam.appearance_models[0]
    SUT_aam.appearance_models.append(App_tmp)

    #===================================================================================================================
    #### Appearance Model Finished ###
    #===================================================================================================================
    ### Bulding the Fitter ###
    #===================================================================================================================
    # Building the Fitter
    fitter =  LucasKanadeAAMFitter(
        SUT_aam,
        n_shape=15,
        n_appearance=100
    )

    #===================================================================================================================
    errors = []
    n_iters = []
    final_errors = []

    # fit images
    for k,i in enumerate(test_images):
        gt_s = i.landmarks['face_ibug_66_trimesh'].lms

        # Loading the perturbations
        with open('/u/azinasg/Research/Sample_UNBC_Small_Init_2/' + i.path.name[:-4] + '.pkl',
                  'rb') as input:
            perturbations = pickle.load(input)

        for j in range(0, 10):
            initial_s = perturbations[j]

            # fit image
            fr = fitter.fit_from_shape(i, initial_s, gt_shape=gt_s,  max_iters=300)
            errors.append(fr.errors())
            n_iters.append(fr.n_iters)
            final_errors.append(fr.final_error())

            print "Ins_ALL_One : alpha=" + str(alpha) + " beta=" + str(beta)+" k=" + str(k) + " j=" + str(j) + \
                  " initial err: " + str(fr.initial_error()) + " final err: " + str(fr.final_error())

    with open(r'/u/azinasg/res16/Instance_All_One_a=' + str(alpha) + '.pkl', 'wb') as f:
        pickle.dump(errors, f)
        pickle.dump(n_iters, f)
        pickle.dump(final_errors, f)
