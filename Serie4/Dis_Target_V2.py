import cPickle as pickle
import copy
import os

import menpo.io as mio
import numpy as np
from menpo.feature import no_op
from menpo.landmark import labeller, face_ibug_68_to_face_ibug_66_trimesh
from menpo.math import as_matrix
from menpo.model import PCAModel
from menpo.transform import PiecewiseAffine
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

    # Swapping Columns - (Y,X) -> (X,Y)
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
### Function for finding effective Source Eigen-Vectors ###
#=======================================================================================================================
def Compute_Effective_Space(T_Data,T_mean,S_Comp):
    N_Samples = T_Data.shape[0]
    N_Comp = S_Comp.shape[0]
    Captured_variance = np.zeros((N_Comp))

    # Mean Centering Target Data
    data = T_Data - T_mean
    for i in range(N_Comp):
        Captured_variance[i] = (1.0/N_Samples) * \
                               np.dot(np.dot(S_Comp[i,:],np.transpose(data)),np.dot(data,np.transpose(S_Comp[i,:])))
    indexes = np.argsort(Captured_variance)[::-1]
    New_Comp = S_Comp[indexes,:]
    Captured_variance = np.sort(Captured_variance)[::-1]
    return New_Comp, Captured_variance

#=======================================================================================================================
### Loading Data ###
#=======================================================================================================================
# Loading the face_ibug_66_trimesh template
with open('/Users/azinasgarian/Documents/Research/face_ibug_66_trimesh_temp.pkl', 'rb') as input:
    land_tmp = pickle.load(input)

# Loading (LFPW-AFW-Helen-Ibug) Images (Source Images)
source_path = Path('/Users/azinasgarian/Documents/Research/Source_Small')
source_images = [load_image(i) for i in mio.import_images(source_path, verbose=True)]

# Loading CK Images (Source Images)
CK_root = "/Users/azinasgarian/Documents/Research/Sample_CK+_Small"
for root, dirs, filenames in os.walk(CK_root):
    for filename in filenames:
        if (".png" in filename) and (".DS_Store" not in filename) :
            tmp_image = Read_CK(land_tmp,root+"/"+filename)
            source_images.append(tmp_image)

# Loading UNBC Images - Target
UNBC_root = "/Users/azinasgarian/Documents/Research/Sample_UNBC_Small_Target"
target_images = []
for root, dirs, filenames in os.walk(UNBC_root):
    for filename in filenames:
        if (".png" in filename) and (".DS_Store" not in filename) :
            target_images.append(Read_UNBC(land_tmp,root+"/"+filename))

# Loading UNBC Images - Test
UNBC_root = "/Users/azinasgarian/Documents/Research/Sample_UNBC_Small_Test_2"
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
### Rescaling Target Images to Target reference shape ###
#=======================================================================================================================
ST = rescale_images_to_reference_shape(target_images, 'face_ibug_66_trimesh', target_aam.reference_shape, verbose=True)

#=======================================================================================================================
### Building Shape Variables ###
#=======================================================================================================================
# Building Target Shape Data Matrix
ST_scaled_shapes = [i.landmarks['face_ibug_66_trimesh'].lms for i in ST]
ST_aligned_shapes = align_shapes(ST_scaled_shapes)
ST_data = as_matrix(ST_aligned_shapes, return_template=False, verbose=True)
ST_N = ST_data.shape[0]
ST_mean = np.mean(ST_data, axis=0)

# Reordering Source Shape vectors based on variance captured in target
New_Shape_Components, New_Shape_Eigen_Values = Compute_Effective_Space(ST_data, ST_mean,
                                                                  source_aam.shape_models[0].model._components)

# Orthonormalize Source Shape Model Against Target Shape Model
source_aam.shape_models[0].model._components = New_Shape_Components
source_aam.shape_models[0].model.orthonormalize_against_inplace(target_aam.shape_models[0].model)

#=======================================================================================================================
### Concatenating Target and Discarded Source Shape Space ###
#=======================================================================================================================
Shape_e_vectors = np.vstack((target_aam.shape_models[0].model._components,source_aam.shape_models[0].model._components))
Shape_e_values = np.concatenate((target_aam.shape_models[0].model._eigenvalues,New_Shape_Eigen_Values))
n_new_samples = ST_N + len(source_aam.shape_models[0].model._eigenvalues)

#=======================================================================================================================
### Updating Shape Components ###
#=======================================================================================================================
Shape_tmp = PCAModel.init_from_components(components=Shape_e_vectors, eigenvalues=Shape_e_values,
                                          mean=SUT_aam.shape_models[0].model.mean(),
                                          n_samples = n_new_samples, centred=True,
                                          max_n_components=SUT_aam.max_shape_components[0])
# Setting Models info
SUT_aam.shape_models[0].model = Shape_tmp

SUT_aam.shape_models[0]._target = None

SUT_aam.shape_models[0]._weights = np.zeros(SUT_aam.shape_models[0].model.n_active_components)

SUT_aam.shape_models[0]._target = SUT_aam.shape_models[0].model.mean()

shape_mean = SUT_aam.shape_models[0].model.mean()

SUT_aam.shape_models[0].global_transform = DifferentiableAlignmentSimilarity(shape_mean,shape_mean)
# Re-orthonormalize components against similarity transfrom
SUT_aam.shape_models[0]._construct_similarity_model()
# Reset the target given the new model
SUT_aam.shape_models[0]._sync_target_from_state()

#=======================================================================================================================
### Building Appearance Model ###
#=======================================================================================================================
# Building Reference frames
SUT_reference_frame = build_reference_frame(SUT_aam.reference_shape)
S_reference_frame = build_reference_frame(source_aam.reference_shape)
T_reference_frame = build_reference_frame(target_aam.reference_shape)

# Obtain warped target samples
ST_warped = warp_images(ST, ST_scaled_shapes, T_reference_frame, target_aam.transform,verbose=True)

# Building Data Matrix
ST_App_data = as_matrix(ST_warped, return_template=False, verbose=True)
ST_App_N = ST_App_data.shape[0]

# Defining the warping from Samples Images to Mean of the Target Images
pwa_s_t = PiecewiseAffine(T_reference_frame.landmarks['source'].lms, S_reference_frame.landmarks['source'].lms)

# Warping Source Appearance Vectors to Target reference frame
Warped_S_to_T = []
for i in range(0, source_aam.appearance_models[0]._components.shape[0]):
    img = S_reference_frame.from_vector(source_aam.appearance_models[0]._components[i, :])
    warped = img.as_unmasked(copy=False).warp_to_mask(T_reference_frame.mask, pwa_s_t)
    Warped_S_to_T.append(warped)
comp_S = as_matrix(Warped_S_to_T, return_template=False, verbose=True)

# Defining Source Weight Vector
New_Appearance_Components, New_Appearance_Eigen_Values = Compute_Effective_Space(ST_App_data,
                                                                           target_aam.appearance_models[0]._mean,
                                                                           comp_S)

# Orthonormalize Source Apearance Model Against Target Shape Model
source_aam.appearance_models[0]._components = New_Appearance_Components
source_aam.appearance_models[0].orthonormalize_against_inplace(target_aam.appearance_models[0])

#=======================================================================================================================
### Building Appearance Model ###
#=======================================================================================================================
# Concatenating Target and Discarded Source Appearance Space
App_e_vectors = np.vstack((target_aam.appearance_models[0]._components,source_aam.appearance_models[0]._components))
App_e_values = np.concatenate((target_aam.appearance_models[0]._eigenvalues, New_Appearance_Eigen_Values))
App_n_new_samples = ST_App_N + len(New_Appearance_Eigen_Values)

# Defining the warping from Target template to SUT template
pwa_t_sut = PiecewiseAffine(SUT_reference_frame.landmarks['source'].lms, T_reference_frame.landmarks['source'].lms)

# Warping Target eigen vectors to SUT reference frame
Warped_T_to_SUT = []
for i in range(0, App_e_vectors.shape[0]):
    img = T_reference_frame.from_vector(App_e_vectors[i, :])
    warped = img.as_unmasked(copy=False).warp_to_mask(SUT_reference_frame.mask, pwa_t_sut)
    Warped_T_to_SUT.append(warped)
App_e_vectors = as_matrix(Warped_T_to_SUT, return_template=False, verbose=True)

#=======================================================================================================================
### Updating Appearance Components and ###
#=======================================================================================================================
App_tmp =  PCAModel.init_from_components(components=App_e_vectors, eigenvalues=App_e_values,
                                          mean=SUT_aam.appearance_models[0].mean(),
                                          n_samples = App_n_new_samples,centred=True,
                                         max_n_components=SUT_aam.max_appearance_components[0])

del SUT_aam.appearance_models[0]
SUT_aam.appearance_models.append(App_tmp)
#=======================================================================================================================
### Looping on Different Values of Alpha ###
#=======================================================================================================================
alpha = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
beta = [13, 30, 47, 65, 83, 100, 118, 135, 153, 170, 187, 200]

for x in range(len(alpha)):
    #===================================================================================================================
    ### Bulding the Fitter ###
    #===================================================================================================================
    fitter =  LucasKanadeAAMFitter(
        SUT_aam,
        n_shape=alpha[x],
        n_appearance=beta[x]
    )

    #===================================================================================================================
    ### Fitting Images ###
    #===================================================================================================================
    errors = []
    n_iters = []
    final_errors = []

    # fitting
    for k, i in enumerate(test_images):
        # Ground Truth
        gt_s = i.landmarks['face_ibug_66_trimesh'].lms

        # Loading the perturbations
        with open('/u/azinasg/Research/Sample_UNBC_Small_Init_2/' + i.path.name[:-4] + '.pkl',
                  'rb') as input:
            perturbations = pickle.load(input)

        for j in range(0, 10):
            initial_s = perturbations[j]

            # fit image
            fr = fitter.fit_from_shape(i, initial_s, gt_shape=gt_s, max_iters=300)
            errors.append(fr.errors())
            n_iters.append(fr.n_iters)
            final_errors.append(fr.final_error())

            print "Dis_T_V2 : A=" + str(alpha[x]) + " B=" + str(beta[x]) + " k=" + str(k) + " j=" + str(j) + \
                  " initial err: " + str(fr.initial_error()) + " final err: " + str(fr.final_error())

    with open(r'/u/azinasg/UNBC_res1/Dis_T_V2_x=' + str(alpha[x]-4) + '.pkl', 'wb') as f:
        pickle.dump(errors, f)
        pickle.dump(n_iters, f)
        pickle.dump(final_errors, f)
