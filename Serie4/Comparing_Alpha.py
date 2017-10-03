from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

# ======================================================================================================================
# Return the tmp Matrix with extended rows with the last element of each row
# ======================================================================================================================
def build_matrix(tmp,nrow,ncol):
    res = np.zeros((nrow,ncol))
    for i in range(0,nrow):
        if len(tmp[i][:]) < ncol :
            res[i][0:len(tmp[i][:])] = tmp[i][:]
            res[i][len(tmp[i][:]):] = tmp[i][len(tmp[i][:])-1]
        else :
            res[i][:] = tmp[i][:ncol]
    return res

# ======================================================================================================================
# Initialization
# ======================================================================================================================
Max_Iter = 300
Sam_Num = 2100
num_plots = 11

errors = []
n_iters = []
final_error = []
Err = []
Err_mean = []
Conv_Err = []
Conv_Err_mean = []
y = []

root = '/Users/azinasgarian/Desktop/res_show/res16/'
Key = ['Instance_All_One_a=', 'Instance_Gauss_a=', 'Instance_Gauss_WL_a=', 'Instance_Heuristic_a=',
       'Instance_Heuristic_WN_a=', 'Instance_Logistic_a=', 'Instance_Logistic_WL_a=', 'SUT_Discarded_Source_x=',
       'T_Discarded_Source_x=', 'SUT_Discarded_Source_V2_x=', 'T_Discarded_Source_V2_x=', 'Azin_a=']

# ======================================================================================================================
### Reading Data
# ======================================================================================================================
# Loading Data
for x in range(num_plots):
    alpha = x/10.0
    with open(r'/Users/azinasgarian/Documents/Research/results/res16/' + Key[4] +
                      str(alpha) + '.pkl', 'rb') as f:
        Temp_errors = pickle.load(f)
        errors.append(Temp_errors)
        Temp_n_iters = np.array(pickle.load(f))
        n_iters.append(Temp_n_iters)
        Temp_final_err = np.array(pickle.load(f))
        final_error.append(Temp_final_err)
        print "Data for alpha = " + str(alpha) + " is readed ! "


# Building Data Matrix - Making the data Symmetric
for i in range(num_plots):
    tmp = build_matrix(errors[i],Sam_Num,Max_Iter+1)
    Err_mean.append(np.mean(tmp,axis=0))
    Err.append(tmp)

# ======================================================================================================================
# Printing Rate of Convergence-Baker Style
# ======================================================================================================================
print "**** Rate of Convergence ****"

for i in range(num_plots):
    f_err = final_error[i]
    n_iter = n_iters[i]
    print "% of Convergence for a=" + str (i/10.0) + " -> Iteration based = " + str(len(n_iter[n_iter < 300]) / (Sam_Num))\
          + " , Error less than 0.05 = " + str(len(f_err[f_err < 0.05]) / (Sam_Num))+"\n"

# ======================================================================================================================
# Plotting Rate of Convergence For Converged Trials-Baker Style
# ======================================================================================================================
# Selecting the converged cases - Which have converged in less than Max_Iter Iterations
for i in range(num_plots):
    tmp_err = Err[i]
    tmp_n_iters = n_iters[i]
    tmp = tmp_err[tmp_n_iters<300][:]
    Conv_Err_mean.append(np.mean(tmp, axis=0))
    Conv_Err.append(tmp)

# Drawing the Plot
plt.figure(2)
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])

for i in range(num_plots):
    plt.plot(np.arange(0, Max_Iter + 1), Conv_Err_mean[i], linestyle='-', linewidth=2, label="a="+str(i/10.0))

plt.xlabel('Iteration')
plt.ylabel('Average BB Normalized RMS Error')
plt.title('Rate of Error for Converged Examples(n_iters<300) - Baker Style')
plt.grid(True)
plt.legend()
# plt.savefig(root + Key + '_1.pdf')
plt.show()


# ======================================================================================================================
# Plotting Image Percentages Vs RMS Error- Menpo Style ######
# ======================================================================================================================

for i in range(num_plots):
    tmp_y=[0]
    tmp_final_error = final_error[i]
    for i in np.arange(0.0005,1.005,0.0005):
        tmp_y.append(len(tmp_final_error[tmp_final_error<=i])/(Sam_Num))
    y.append(tmp_y)

# Drawing the Plot
plt.figure(3)
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
x = np.arange(0.000,1.005,0.0005)
for i in range(num_plots):
    plt.plot(x,y[i], linestyle='-', linewidth=2, label="a="+str(i/10.0))

plt.xlabel('BB_Normalized_RMS_Error')
plt.ylabel('Image_Percentages')
plt.title('Images Percentages Vs BB Normalized RMS Error')
plt.grid(True)
plt.legend()
# plt.savefig(root + Key + '_2.pdf')
plt.show()

