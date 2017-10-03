from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import itertools

# ======================================================================================================================
# Return the tmp matrix with extended rows with the last element of each row
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
errors = []
n_iters = []
final_error = []
Max_Iter = 300
Sam_Num = 2100

Err = []
Err_mean = []
Conv_Err = []
Conv_Err_mean = []
y = []
# all_one_0.9 check !
# Names = ['T_Only', 'S_Only', 'SUT', 'Instance_Gauss_WL_a=0.9', 'Instance_Heuristic_a=0.9',
#          'Instance_Heuristic_WN_a=0.5', 'Instance_Logistic_a=0.9', 'Instance_Logistic_WL_a=0.6',
#          'SUT_Discarded_Source_x=2', 'T_Discarded_Source_x=2', 'SUT_Discarded_Source_V2_x=1', 'T_Discarded_Source_V2_x=1']

# Normalized Heu is used !
Names = ['T_Only', 'S_Only', 'SUT', 'Dis_S_V2_x=11', 'Instance_Heuristic_WN_a=0.5', 'SUT_Discarded_Source_V2_x=1']
# Lables = ['T_Only', 'S_Only', 'SUT', 'Ins_Gauss', 'Ins_Hue', 'Ins_Heuristic_WN',
#           'Ins_Logistic','Ins_Logistic_WL', 'SUT_Dis', 'T_Dis', 'SUT_Dis_V2', 'T_Dis_V2']
Lables = ['Target (T)', 'Source (S)', 'Union of S & T (SUT)', 'Subspace Transfer', 'Instance-Weighted [11]', 'Ours']

marker = itertools.cycle(('>', '<', 'o', 'd', 'v', '*'))
stride = 10

num_plots = len(Lables)

# ======================================================================================================================
# Reading data
# ======================================================================================================================
for name in Names:
    with open(r'/Users/azinasgarian/Documents/Research/results/res16/' + name + '.pkl', 'rb') as f:
        Temp_errors = pickle.load(f)
        errors.append(Temp_errors)
        Temp_n_iters = np.array(pickle.load(f))
        n_iters.append(Temp_n_iters)
        Temp_final_err = np.array(pickle.load(f))
        final_error.append(Temp_final_err)
        print "Data for " + name + " is readed ! "


# Building data matrix - making data matrix symmetric
for i in range(num_plots):
    tmp = build_matrix(errors[i],Sam_Num,Max_Iter+1)
    Err_mean.append(np.mean(tmp,axis=0))
    Err.append(tmp)

# ======================================================================================================================
# Printing rate of convergence-baker style
# ======================================================================================================================
print "**** Rate of Convergence ****"

for i in range(num_plots):
    f_err = final_error[i]
    iter = n_iters[i]
    print "% of Convergence for " + Lables[i] + " -> Iteration based = " + str(len(iter[iter < 300]) / (Sam_Num)) + \
          " , Error less than 0.05 = " + str(len(f_err[f_err < 0.05]) / (Sam_Num))+"\n"

# ======================================================================================================================
# Plotting rate of convergence for converged trials - Baker Style
# ======================================================================================================================
# Selecting the converged cases - Which have converged in less than Max_Iter Iterations
for i in range(num_plots):
    tmp_err = Err[i]
    tmp_n_iters = n_iters[i]
    tmp = tmp_err[tmp_n_iters<Max_Iter][:]
    Conv_Err_mean.append(np.mean(tmp, axis=0))
    Conv_Err.append(tmp)

# ======================================================================================================================
# Drawing the plot
# ======================================================================================================================
plt.figure(2)
colormap = plt.cm.hsv
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0.1, 0.9, num_plots)])

for i in range(num_plots):
    plt.plot(np.arange(0, Max_Iter + 1), Conv_Err_mean[i], linestyle='-', marker=marker.next() , linewidth=2,
             label=Lables[i], markevery = stride)

plt.xlabel('Iteration')
plt.ylabel('Average BB Normalized RMS Error')
plt.title('RMS error  for Converged Examples (n_iters<300) - Baker Style')
plt.grid(True)
plt.legend()
plt.show()

# ======================================================================================================================
# Plotting rate of convergence for converged to optimal slution trials - Baker Style
# ======================================================================================================================
# Selecting the converged cases - Which have converged in less than Max_Iter Iterations
Conv_Err_mean = []
Conv_Err = []
for i in range(num_plots):
    tmp_err = Err[i]
    tmp_final_error = final_error[i]
    tmp = tmp_err[tmp_final_error<0.05][:]
    Conv_Err_mean.append(np.mean(tmp, axis=0))
    print Lables[i] + '  mean = ' + str(np.mean(tmp, axis=0)[300])
    Conv_Err.append(tmp)

# ======================================================================================================================
# Drawing the plot
# ======================================================================================================================
plt.figure(3)
colormap = plt.cm.hsv
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0.1, 0.9, num_plots)])

for i in range(num_plots):
    plt.plot(np.arange(0, Max_Iter + 1), 100.0 * Conv_Err_mean[i], linestyle='-', marker=marker.next(), linewidth=2,
             label=Lables[i], markevery = stride)

plt.xlabel('Iteration')
plt.ylabel('Mean RMS Error (% face size)')
#plt.title('RMS error for Converged Examples (error<0.05)')
plt.grid(True)
plt.legend()
plt.savefig('/Users/azinasgarian/Desktop/acc-a.pdf', bbox_inches='tight')
plt.show()

# ======================================================================================================================
# Plotting Image Percentages Vs RMS Error- Menpo Style
# ======================================================================================================================
for i in range(num_plots):
    tmp_y=[0]
    tmp_final_error = final_error[i]
    for i in np.arange(0.0005,1.005,0.0005):
        tmp_y.append(len(tmp_final_error[tmp_final_error<=i])/(Sam_Num))
    y.append(tmp_y)

# Drawing the Plot
plt.figure(4)
stride = 2
colormap = plt.cm.hsv
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0.1, 0.9, num_plots)])
x = np.arange(0.000,1.005,0.0005)*100
for i in range(0,num_plots,1):
    plt.plot(x, 100 * np.array(y[i]), linestyle='-', marker=marker.next(), linewidth=2, label=Lables[i], markevery = stride)

plt.xlabel('RMS Error (% face size)')
plt.ylabel('Test Examples Converged (%)')
#plt.title('Images Percentages Vs BB Normalized RMS Error')
plt.grid(True)
plt.xlim(1.5,5)
plt.ylim(0,90)
plt.legend(loc=2)
plt.savefig('/Users/azinasgarian/Desktop/conv-a.pdf', bbox_inches='tight')
plt.show()
