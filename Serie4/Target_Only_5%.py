import cPickle as pickle
import copy
import os

import menpo.io as mio
import numpy as np
from menpo.feature import no_op
from menpo.landmark import labeller, face_ibug_68_to_face_ibug_66_trimesh
from menpofit.aam import HolisticAAM, LucasKanadeAAMFitter


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
### Pre-Computation ###
#=======================================================================================================================
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

target_fitter =  LucasKanadeAAMFitter(
    target_aam,
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
    with open('/u/azinasg/Research/Sample_UNBC_Small_Init_2/' + i.path.name[:-4] + '.pkl',
              'rb') as input:
        perturbations = pickle.load(input)

    for j in range(0, 10):
        initial_s = perturbations[j]

        # fit image
        fr = target_fitter.fit_from_shape(i, initial_s, gt_shape=gt_s,  max_iters=300)
        errors.append(fr.errors())
        n_iters.append(fr.n_iters)
        final_errors.append(fr.final_error())

        print "T_Only : k=" + str(k) + " j=" + str(j) + " initial err: " + \
              str(fr.initial_error()) + " final err: " + str(fr.final_error())

with open(r'/u/azinasg/res16/T_Only.pkl', 'wb') as f:
    pickle.dump(errors, f)
    pickle.dump(n_iters, f)
    pickle.dump(final_errors, f)
