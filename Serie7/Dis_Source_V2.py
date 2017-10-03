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
    i = i.crop_to_landmarks_proportion(0.1)
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
    img = img.crop_to_landmarks_proportion(0.1)
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
    img = img.crop_to_landmarks_proportion(0.1)
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

# Loading CK+ Images - Target
CK_root = "/u/azinasg/Research/CK+_Target_2"
target_images = []
for root, dirs, filenames in os.walk(CK_root):
    for filename in filenames:
        if (".png" in filename) and (".DS_Store" not in filename) :
            target_images.append(Read_CK(land_tmp,root+"/"+filename))

# Loading CK+ Images - Test
CK_root = "/u/azinasg/Research/CK+_Test_3"
test_images = []
for root, dirs, filenames in os.walk(CK_root):
    for filename in filenames:
        if (".png" in filename) and (".DS_Store" not in filename):
            test_images.append(Read_CK(land_tmp, root + "/" + filename))

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
source_aam.shape_models[0].model.orthonormalize_against_inplace(target_aam.shape_models[0].model)

#=======================================================================================================================
### Concatenating Target and Discarded Source Shape Space ###
#=======================================================================================================================
Shape_e_vectors = np.vstack((target_aam.shape_models[0].model._components,source_aam.shape_models[0].model._components))
Shape_e_values = np.concatenate((target_aam.shape_models[0].model._eigenvalues,
                                 source_aam.shape_models[0].model._eigenvalues))
n_new_samples = len(target_images) + len(source_images)

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

# Defining the warping from Samples Images to Mean of the Target Images
pwa_s_t = PiecewiseAffine(T_reference_frame.landmarks['source'].lms, S_reference_frame.landmarks['source'].lms)

# Warping Source Appearance Vectors to Target reference frame
Warped_S_to_T = []
for i in range(0, source_aam.appearance_models[0]._components.shape[0]):
    img = S_reference_frame.from_vector(source_aam.appearance_models[0]._components[i, :])
    warped = img.as_unmasked(copy=False).warp_to_mask(T_reference_frame.mask, pwa_s_t)
    Warped_S_to_T.append(warped)
comp_S = as_matrix(Warped_S_to_T, return_template=False, verbose=True)

# Orthonormalize Source Apearance Model Against Target Shape Model
source_aam.appearance_models[0]._components = comp_S
source_aam.appearance_models[0].orthonormalize_against_inplace(target_aam.appearance_models[0])

#=======================================================================================================================
### Building Appearance Model ###
#=======================================================================================================================
# Concatenating Target and Discarded Source Appearance Space
App_e_vectors = np.vstack((target_aam.appearance_models[0]._components,source_aam.appearance_models[0]._components))
App_e_values = np.concatenate((target_aam.appearance_models[0]._eigenvalues,
                               source_aam.appearance_models[0]._eigenvalues))
App_n_new_samples = n_new_samples

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
alpha = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
beta = [4, 13, 21, 30, 39, 47, 56, 65, 74, 83, 91, 100, 109, 118, 126, 135, 144, 153, 161, 170, 178, 187, 196, 200]

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
        with open('/u/azinasg/Research/CK+_Init_3/' + i.path.name[:-4] + '.pkl',
                  'rb') as input:
            perturbations = pickle.load(input)

        for j in range(0, 10):
            initial_s = perturbations[j]

            # fit image
            fr = fitter.fit_from_shape(i, initial_s, gt_shape=gt_s, max_iters=300)
            errors.append(fr.errors())
            n_iters.append(fr.n_iters)
            final_errors.append(fr.final_error())

            print "Dis_S_V2 : A=" + str(alpha[x]) + " B=" + str(beta[x]) + " k=" + str(k) + " j=" + str(j) + \
                  " initial err: " + str(fr.initial_error()) + " final err: " + str(fr.final_error())

    with open(r'/u/azinasg/CK+_res4/SUT_Discarded_Source_V2_x=' + str(alpha[x]-4) + '.pkl', 'wb') as f:
        pickle.dump(errors, f)
        pickle.dump(n_iters, f)
        pickle.dump(final_errors, f)
