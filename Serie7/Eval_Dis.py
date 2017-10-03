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
Sam_Num = 1500

Err = []
Err_mean = []
Conv_Err = []
Conv_Err_mean = []
y = []

# Normalized Heu is used !
Names = ['Dis_T_V2_x=0', 'Dis_T_V2_x=1', 'Dis_T_V2_x=2', 'Dis_T_V2_x=3', 'Dis_T_V2_x=4', 'Dis_T_V2_x=5', 'Dis_T_V2_x=6',
         'Dis_T_V2_x=7', 'Dis_T_V2_x=8', 'Dis_T_V2_x=9', 'Dis_T_V2_x=10', 'Dis_T_V2_x=11' ,'Dis_T_V2_x=12',
         'Dis_T_V2_x=13', 'Dis_T_V2_x=14' , 'Dis_T_V2_x=15', 'Dis_T_V2_x=16', 'Dis_T_V2_x=17', 'Dis_T_V2_x=18',
          'Dis_T_V2_x=20', 'Dis_S_V2_x=0', 'Dis_S_V2_x=1', 'Dis_S_V2_x=2', 'Dis_S_V2_x=3',
         'Dis_S_V2_x=4', 'Dis_S_V2_x=5', 'Dis_S_V2_x=6', 'Dis_S_V2_x=7', 'Dis_S_V2_x=8', 'Dis_S_V2_x=9',
         'Dis_S_V2_x=10', 'Dis_S_V2_x=11', 'Dis_S_V2_x=12', 'Dis_S_V2_x=13', 'Dis_S_V2_x=14', 'Dis_S_V2_x=15',
         'Dis_S_V2_x=16', 'Dis_S_V2_x=17', 'Dis_S_V2_x=18', 'Dis_S_V2_x=20']

Lables = ['Ours-0', 'Ours-1', 'Ours-2', 'Ours-3', 'Ours-4', 'Ours-5' ,'Ours-6', 'Ours-7', 'Ours-8', 'Ours-9', 'Ours-10',
          'Ours-11', 'Ours-12', 'Ours-13', 'Ours-14', 'Ours-15', 'Ours-16', 'Ours-17', 'Ours-18', 'Ours-20', 'Dis-S-0',
          'Dis-S-1', 'Dis-S-2', 'Dis-S-3', 'Dis-S-4', 'Dis-S-5', 'Dis-S-6', 'Dis-S-7', 'Dis-S-8', 'Dis-S-9', 'Dis-S-10',
          'Dis-S-11', 'Dis-S-12', 'Dis-S-13', 'Dis-S-14', 'Dis-S-15', 'Dis-S-16', 'Dis-S-17', 'Dis-S-18',
          'Dis-S-20']

marker = itertools.cycle(('o'))
stride = 20

num_plots = len(Lables)

# ======================================================================================================================
# Reading data
# ======================================================================================================================
for name in Names:
    with open(r'/Users/azinasgarian/Documents/Research/results/CK+_res4/' + name + '.pkl', 'rb') as f:
        Temp_errors = pickle.load(f)
        errors.append(Temp_errors)
        Temp_n_iters = np.array(pickle.load(f))
        n_iters.append(Temp_n_iters)
        Temp_final_err = np.array(pickle.load(f))
        final_error.append(Temp_final_err)
        print "Data for " + name + " is read ! "


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

print 'Step 1'
# ======================================================================================================================
# Drawing the plot
# ======================================================================================================================
# plt.figure(2)
# colormap = plt.cm.hsv
# plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0.1, 0.9, num_plots)])
#
# for i in range(num_plots):
#     plt.plot(np.arange(0, Max_Iter + 1), Conv_Err_mean[i], linestyle='-', marker=marker.next() , linewidth=2,
#              label=Lables[i], markevery = stride)
#
# plt.xlabel('Iteration')
# plt.ylabel('Average BB Normalized RMS Error')
# plt.title('RMS error  for Converged Examples (n_iters<300) - Baker Style')
# plt.grid(True)
# plt.legend()
# plt.show()

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

print 'Step 2'
# ======================================================================================================================
# Drawing the plot
# ======================================================================================================================
# plt.figure(3)
# colormap = plt.cm.hsv
# plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0.1, 0.9, num_plots)])
#
# for i in range(num_plots):
#     plt.plot(np.arange(0, Max_Iter + 1), Conv_Err_mean[i], linestyle='-', marker=marker.next(), linewidth=2,
#              label=Lables[i], markevery = stride)
#
# plt.xlabel('Iteration')
# plt.ylabel('Normalized RMS error by face size')
# #plt.title('RMS error for Converged Examples (error<0.05)')
# plt.grid(True)
# plt.legend()
# plt.show()

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
# plt.figure(4)
# colormap = plt.cm.hsv
# plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0.1, 0.9, num_plots)])
# x = np.arange(0.000,1.005,0.0005)
# for i in range(0,num_plots,1):
#     plt.plot(x,y[i], linestyle='-', marker=marker.next(), linewidth=2, label=Lables[i], markevery = stride)
#
# plt.xlabel('Normalized RMS error by face size')
# plt.ylabel('Image Percentages')
# #plt.title('Images Percentages Vs BB Normalized RMS Error')
# plt.grid(True)
# #plt.xlim(0.015,0.05)
# #plt.ylim(0,0.90)
# plt.legend(loc=2)
# plt.show()


# ======================================================================================================================
# Plotting Source and Target
# ======================================================================================================================
y_T = []
y_S = []
tresh = 0.05
for k in range(20):
    y_T.append(y[k][99])

for k in range(20,len(Names)):
    y_S.append(y[k][99])

# Drawing the Plot
plt.figure(5)
colormap = plt.cm.hsv
plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0.1, 0.9, 2)])
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20]

plt.plot(x, y_S, linestyle='-', marker='d', linewidth=3, label='Default Ordering')
plt.plot(x, y_T, linestyle='-', marker='d', linewidth=3, label='Ordering Based on Proposed Metric')

plt.xlabel('Number of Shape Components')
plt.ylabel('Proportion of Test Examples Converged')
#plt.title('Images Percentages Vs BB Normalized RMS Error')
plt.grid(True)
# plt.xlim(0.015,0.05)
plt.ylim(0,1)
plt.legend(loc=1)
plt.savefig('/Users/azinasgarian/Desktop/T&S.pdf', bbox_inches='tight')
plt.show()

print 'hi'
print 'hello'




