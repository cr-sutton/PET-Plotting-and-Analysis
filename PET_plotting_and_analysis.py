# -*- coding: utf-8 -*-
"""
threeD_data_plots
Created on Thu Apr  8 17:15:04 2021
@author: Christopher Zahasky (czahasky@wisc.edu)
"""

# import necessary packages
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import rc

# import filters for 'ridge' detection
from skimage import filters

from scipy.special import erfc as erfc
from scipy.integrate import trapezoid as trapz

rc('font',**{'family':'serif','serif':['Arial']})
fs = 14
plt.rcParams['font.size'] = fs

# import our useful functions
from plotting_and_analysis_functions import plot_2d, plot_3d, quantile_map_function, diffusion_fun, diffusion_integration_fun


# Import arrival time data
data_path = 'C:\\Users\\crsutton\\Documents\\cPET3D_tracer_preacid_nan'

# data_dir_arrival = os.path.join('.')
# Import data
import_data = np.loadtxt(data_path + '.csv', delimiter=',')

# extract metadata
nrow = int(import_data[-8])
ncol = int(import_data[-7])
nslice = int(import_data[-6])
ntimes = int(import_data[-5]) 
timestep_size = import_data[-4] 
dz = import_data[-1] # voxel size in z direction (parallel to axis of core)
dy = import_data[-2]  # voxel size in y direction
dx = import_data[-3]  # voxel size in x direction

import_data = import_data[0:-8]
PET_data = import_data.reshape(nrow, ncol, nslice, ntimes)
# crop empty voxels on left and top
PET_data = PET_data[:-1, 1:,:,:]
nrow -=1
ncol -=1

# plot example slices at a given timestep
timestep = 5
PET_frame = PET_data[:,:,:, timestep]

plot_2d(PET_frame[:,:,15], dx, dy, 'Radiotracer concentration', 'Reds')
plot_2d(PET_frame[:,15,:], dz, dy, 'Radiotracer concentration', 'Reds')
plot_2d(PET_frame[15,:,:], dz, dy, 'Radiotracer concentration', 'Reds')

timesteps_for_change = 5 # this was chosen from arrival time range
dcdt_matrix = np.mean((PET_data[:,:,:, 1:timesteps_for_change+1]-PET_data[:,:,:, 0:timesteps_for_change])/timestep_size, 3)

# plot example slice of this matrix of rates of change
test_slice = 15
plot_2d(dcdt_matrix[:,:,test_slice], dx, dy, 'Concentration change', 'viridis')

# let's swap out the nans for zeros in the change matrix
dcdt_matrix[np.isnan(dcdt_matrix[:,:,:])] = 0
# add zeros to all sides to catch fractures near core edge
padded_dcdt = np.pad(dcdt_matrix, pad_width=1)

# loop through slices and find ridge using skimage.filters
# preallocate ridge matrix
ridge_matrix = np.zeros([nrow+2, ncol+2, nslice+2])
for i in range(0, nslice):
    ridge_matrix[:,:,i] = filters.frangi(padded_dcdt[:,:,i], scale_range =[0.1, 1.0], scale_step = 0.1, black_ridges=False)

# crop pads
ridge_matrix = ridge_matrix[1:-1, 1:-1, 1:-1]
ridge_matrix.shape
# crop out any edge artificts
ridge_matrix[np.isnan(PET_frame[:,:,:])] = 0

# segment location of fracture
binary_frac_matrix = (ridge_matrix> np.max(ridge_matrix*0.1))*1

plot_2d(ridge_matrix[:,:,test_slice], dx, dy, 'frangi', 'viridis')
plt.title('example slice showing ridge filter results')

plot_2d(binary_frac_matrix[:,:,test_slice], dx, dy, 'segmented frangi', 'gray_r')
plt.title('example slice showing fracture location')


#### plot breakthrough curves of voxels in fracture (as determined by binary matrix)
plt.figure(dpi=200)
# define time array based on image data
time_array = time_array = np.linspace(timestep_size/2, timestep_size*ntimes, ntimes, endpoint=True)
# time_array = np.linspace(timestep_size, timestep_size*plot_end_time, plot_end_time, endpoint=True)
# determine number of line colors

colors = plt.cm.inferno(np.linspace(0,1, np.sum(binary_frac_matrix)))
n=0
for i in range(0,nrow):
    for j in range(0,ncol):
        for k in range(0, nslice):
            # if fracture is present, plot btc
            if binary_frac_matrix[i,j,k]==1:
                btc = PET_data[i,j,k,:]
                plt.plot(time_array, btc, color=colors[n], linewidth=0.1, alpha = 0.6)
                n +=1
plt.xlabel('Time [sec]')
plt.ylabel('Concentration')
plt.title('breakthrough curves of fracture voxels')

quantile_map = quantile_map_function(PET_data, timestep_size, [dx, dy, dz], 0.01)
# remove edges
quantile_frac_map = quantile_map*binary_frac_matrix

# now swap values out with nans again so that they are easier to plot in slices
quantile_frac_map[quantile_frac_map<1]=np.nan

# optional: offset arrival time so that it starts from zero
# quantile_frac_map = quantile_frac_map - np.nanmin(quantile_frac_map)


####### slice plots
# # Plot subplots of slices
r, c = np.shape(PET_frame[:,:,1])
x_coord = np.linspace(0, dx*c, c)
y_coord = np.linspace(0, dy*r, r)

X, Y = np.meshgrid(x_coord, y_coord)

fig, axs = plt.subplots(3, 4, sharex=True, sharey=True, dpi=200, figsize=(12, 7))
n=0
vmax_color = np.nanmax(quantile_frac_map)
vmin_color = np.nanmin(quantile_frac_map)

for ax in axs.flat:
    im = ax.pcolormesh(X, Y, quantile_frac_map[:,:,round(n*4)], cmap=plt.cm.viridis, 
                       vmin = vmin_color, vmax = vmax_color, shading = 'auto', edgecolor ='none', linewidth = 0.01, alpha=1.0)
    
    ax.set_title('Slice = {0}'.format(n*4))
    ax.set_aspect('equal')
    n += 1

cax = plt.axes([0.9, 0.1, 0.02, 0.8])
plt.colorbar(im, cax=cax, label = 'Arrival time [sec]')


### frac location
plot_matrix = PET_frame

fig, axs = plt.subplots(3, 4, sharex=True, sharey=True, dpi=200, figsize=(12, 7))
# cont_values = np.arange(0, np.nanmax(PET_frame), 0.005)
n=0
for ax in axs.flat:
    # Plot concentration map
    im = ax.contourf(X, Y, plot_matrix[:,:,round(n*4)], cmap=plt.cm.Reds)
    # overlay fracture location
    im2 = ax.pcolormesh(X, Y, binary_frac_matrix[:,:,round(n*4)], cmap=plt.cm.gray_r, shading = 'auto', edgecolor ='none', alpha=0.3)
    ax.set_title('Slice = {0}'.format(n*4))
    ax.set_aspect('equal')
    n += 1

cax = plt.axes([0.9, 0.1, 0.02, 0.8])
plt.colorbar(im, cax=cax, label = 'Radiotracer Concentration')
plt.show()

# 3d plot (kinda sucks)
# quantile_frac_map[np.isnan(quantile_frac_map)]=0
# plot_3d(quantile_frac_map, dx, dy, dz, climit=[vmin_color, vmax_color])

#function to integrate c equation 
def diffusion_integration_fun_v2(btcArrayShort, dx, t_short, Df_Sim, C0_Sim):
    # Equation for concentration profile as a function of space (x) and time (t)
    xsim = np.linspace(0,dx, num=29) # x in meters
    #xsim = x_coord
    M1d_sim = np.zeros(len(t_short))
    for i in range(0, len(t_short)):
        Csim = C0_Sim*(erfc((xsim)/(2*np.sqrt(Df_Sim*t_short[i]))))
        M1d_sim[i] = trapz(Csim, xsim)
    
    return M1d_sim

#iterative process to fit arrival time, df, and c0 to the PET break through curves
def fit_btc(btcArray, dx,  t_PETSim, Df_Sim, C0_Sim):
    rmse_sim = np.ones([len(Df_Sim), len(C0_Sim), len(t_arrival)])
    for jsim in range(0, len(Df_Sim)):
        # print(jsim)
        for ksim in range(0, len(C0_Sim)):
            # print(ksim)
            for lsim in range(0, len(t_arrival)):
                # print(lsim)
                t_short = t_PETSim - t_arrival[lsim]
                btcArrayShort = btcArray[t_short>0]
                t_short = t_short[t_short>0]
                M1d_sim = diffusion_integration_fun_v2(btcArrayShort, dx/2, t_short, Df_Sim[jsim], C0_Sim[ksim])
                M1dvox = M1d_sim*2
                MSE = (btcArrayShort-M1dvox)**2
                RMSE = np.mean(MSE)
                #print(RMSE)
                rmse_sim[jsim,ksim,lsim] = RMSE
                            
                #find the index of the minimum RMSE
                indices =  np.argpartition(rmse_sim.flatten(), 0)[0]
                ind = np.unravel_index(indices, rmse_sim.shape)
                # Df_Soln[r, c, s] = Df_Sim[ind[0]]
                # C0_Soln[r, c, s] = C0_Sim[ind[1]]
                # tArriv_Soln[r, c, s] = t_arrival[ind[2]]
                
    return Df_Sim[ind[0]], C0_Sim[ind[1]], t_arrival[ind[2]], rmse_sim

from datetime import datetime
start_time = datetime.now()

t_PETSim = time_array # t in seconds
C0_Sim=np.linspace(1, 20, num=40)
Df_Sim = np.logspace(-2, -8, num=50) # in cm^2/s
t_arrival = np.linspace(10, 1000, num=30) # in cm^2/s
Df_Soln = np.zeros([nrow-1, ncol-1, nslice-1])
C0_Soln = np.zeros([nrow-1, ncol-1, nslice-1])
tArriv_Soln = np.zeros([nrow-1, ncol-1, nslice-1])

for r in range(0,5):
    #print(r)
    for c in range(0,ncol):
        #print(c)
        for s in range(0, nslice):
            #print(s)
            # if fracture is present, plot btc
            if binary_frac_matrix[r,c,s]==1:     
                btcArray = PET_data[r, c, s, :]
                dfit, cfit, tfit, rmse_sim = fit_btc(btcArray, dx/2, t_PETSim, Df_Sim, C0_Sim)
                Df_Soln[r, c, s] = dfit
                C0_Soln[r, c, s] = cfit
                tArriv_Soln[r, c, s] = tfit
                plt.plot(t_PETSim, btcArray, linewidth=1, alpha = 1)
                M1d_sim = diffusion_integration_fun_v2(btcArray, dx/2, t_PETSim, dfit, cfit)
                plt.plot(t_PETSim+tfit, M1d_sim*2, linewidth=1, alpha = 1)
                
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))         
              
plt.xlabel('Time [sec]')
plt.ylabel('Concentration')
plt.title('breakthrough curves of fracture voxels')


plot_2d(rmse_sim[:,:,0], 1, 1, 'rmse', 'viridis')
  
plot_2d(Df_Soln[:,:,20], dx, dy, 'df sim slice', 'viridis')
plot_2d(C0_Soln[:,:,20], dx, dy, 'C0 sim slice', 'viridis')
plot_2d(tArriv_Soln[:,:,20], dx, dy, 'Arrival time sim slice', 'viridis')

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

plt.plot(t_PETSim, btcArray, linewidth=1, alpha = 1)
M1d_sim = diffusion_integration_fun_v2(btcArray, dx/2, t_PETSim, 0.0000001, 8)
plt.plot(t_PETSim+500, M1d_sim*2, linewidth=1, alpha = 1)