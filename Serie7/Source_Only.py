import cPickle as pickle
import copy
import os

import menpo.io as mio
import numpy as np
from menpo.feature import no_op
from menpo.landmark import labeller, face_ibug_68_to_face_ibug_66_trimesh
from menpofit.aam import HolisticAAM, LucasKanadeAAMFitter
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

# Building the Fitter
source_fitter =  LucasKanadeAAMFitter(
        source_aam,
        n_shape=15,
        n_appearance=100
    )


#=======================================================================================================================
### Fitting ###
#=======================================================================================================================
errors = []
n_iters = []
final_errors = []

for k,i in enumerate(test_images):
    gt_s = i.landmarks['face_ibug_66_trimesh'].lms

    # Loading the perturbations
    with open('/u/azinasg/Research/CK+_Init_3/' + i.path.name[:-4] + '.pkl',
              'rb') as input:
        perturbations = pickle.load(input)

    for j in range(0, 10):
        initial_s = perturbations[j]

        # fit image
        fr = source_fitter.fit_from_shape(i, initial_s, gt_shape=gt_s,  max_iters=300)
        errors.append(fr.errors())
        n_iters.append(fr.n_iters)
        final_errors.append(fr.final_error())

        print "Source_Only : k=" + str(k) + " j=" + str(j) + " initial err: " + \
              str(fr.initial_error()) + " final err: " + str(fr.final_error())

with open(r'/u/azinasg/CK+_res3/S_Only.pkl', 'wb') as f:
    pickle.dump(errors, f)
    pickle.dump(n_iters, f)
    pickle.dump(final_errors, f)
