# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 21:11:07 2021
@author: Czahasky
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time 

from scipy import integrate
from scipy.special import erfc as erfc
from scipy.integrate import trapezoid as trapz

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Arial']})
fs = 14
plt.rcParams['font.size'] = fs

# Much faster quantile calculation 
def quantile_calc(btc_1d, timearray, quantile):
    # calculate cumulative amount of solute passing by location
    M0i = integrate.cumtrapz(btc_1d, timearray)
    # normalize by total to get CDF
    quant = M0i/M0i[-1]
    # calculate midtimes
    mid_time = (timearray[1:] + timearray[:-1]) / 2.0
    
    # now linearly interpolate to find quantile
    gind = np.argmax(quant > quantile)
    m = (quant[gind] - quant[gind-1])/(mid_time[gind] - mid_time[gind-1])
    b = quant[gind-1] - m*mid_time[gind-1]
    
    tau = (quantile-b)/m
    return tau
            
# Function to calculate the quantile arrival time map
def quantile_map_function(conc, timestep, grid_size, quantile):
    # start timer
    tic = time.perf_counter()
    
    # determine the size of the data
    conc_size = conc.shape
    
    # define array of times based on timestep size (in seconds)
    # Note that we are referencing the center of the imaging timestep since a 
    # given frame is an average of the detected emission events during that period
    timearray = np.arange(timestep/2, timestep*conc_size[3], timestep)
    
    # sum of slice concentrations for calculating inlet and outlet breakthrough
    oned = np.nansum(np.nansum(conc, 0), 0)
    
    # arrival time calculation in inlet slice
    tau_in = quantile_calc(oned[0,:], timearray, quantile)
    
    # arrival time calculation in outlet slice
    tau_out = quantile_calc(oned[-1,:], timearray, quantile)

    # core length
    core_length = grid_size[2]*conc_size[2]
    # array of grid cell centers before interpolation
    z_coord_pet = np.arange(grid_size[2]/2, core_length, grid_size[2])
    
    # Preallocate arrival time array
    at_array = np.zeros((conc_size[0], conc_size[1], conc_size[2]), dtype=float)
    
    for xc in range(0, conc_size[0]):
        for yc in range(0, conc_size[1]):
            for zc in range(0, conc_size[2]):
                # Check if outside core
                if np.isfinite(conc[xc, yc, zc, 0]):
                    # extract voxel breakthrough curve
                    cell_btc = conc[xc, yc, zc, :]
                    # check to make sure tracer is in grid cell
                    if cell_btc.sum() > 0:
                        # call function to find quantile of interest
                        at_array[xc, yc, zc] = quantile_calc(cell_btc, timearray, quantile)
    
    # Replace nans with zeros
    at_array[np.isnan(conc[:,:,:,0])] = 0

    # stop timer
    toc = time.perf_counter()
    print(f"Function runtime is {toc - tic:0.4f} seconds")
    return at_array


# Function for 2D plots
def plot_2d(map_data, dx, dy, colorbar_label, cmap):
    fs = 14
    r, c = np.shape(map_data)
    # Define grid    
    x_coord = np.linspace(0, dx*c, c+1)
    y_coord = np.linspace(0, dy*r, r+1)
    
    X, Y = np.meshgrid(x_coord, y_coord)
    
    # Use 'pcolor' function to plot 2d map of concentration
    plt.figure(figsize=(12, 4), dpi=200)
    plt.pcolormesh(X, Y, map_data, cmap=cmap, shading = 'auto', edgecolor ='k', linewidth = 0.01)
    plt.gca().set_aspect('equal')  
    # add a colorbar
    cbar = plt.colorbar() 
    # plt.clim(cmin, cmax) 
    # label the colorbar
    cbar.set_label(colorbar_label)
    # make colorbar font bigger
    # cbar.ax.tick_params(labelsize= (fs-2)) 
    # make axis fontsize bigger!
    # plt.tick_params(axis='both', which='major', labelsize=fs)
    plt.xlim((0, dx*c)) 
    plt.ylim((0, dy*r)) 
    
    
# functions for 3D plots
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
def half_core(data):
    nrow, c, nslice = np.shape(data)
    data = data[:,:-round(c/2),:]
    ncol = round(c/2)
    return data, nrow, ncol, nslice


def plot_3d(map_data, dx, dy, dz, climit):
    # crop core
    # map_data, nrow, ncol, nslice = half_core(map_data)
    nrow, ncol, nslice = np.shape(map_data)
    
    # swap axes
    map_data = np.flip(map_data, 0)
    map_data = np.swapaxes(map_data,0,2)
    
    # generate grid    
    X, Y, Z = np.meshgrid(np.linspace(dy/2, (ncol-2)*dy+dy/2, num=(ncol+1)), \
                          np.linspace(dz/2, (nslice-2)*dz+dz/2, num=(nslice+1)), \
                          np.linspace(dx/2, (nrow-2)*dx+dx/2, num=(nrow+1)))
    
    
    angle = -30
    fig = plt.figure(figsize=(12, 9), dpi=300)
    ax = fig.gca(projection='3d')
    ax.view_init(30, angle)
    # ax.set_aspect('equal') 
    
    # if color bar range is not defined then find min and max
    if climit == 0:
        norm = matplotlib.colors.Normalize(vmin=map_data.min().min(), vmax=map_data.max().max())
    else:
        norm = matplotlib.colors.Normalize(vmin=climit[0], vmax=climit[1])
        
    # ax.voxels(filled, facecolors=facecolors, edgecolors='gray', shade=False)
    ax.voxels(X, Y, Z, map_data, facecolors=plt.cm.viridis(norm(map_data)), \
              edgecolors='grey', linewidth=0.2, shade=True, alpha=0.7)
    
    m = cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    m.set_array([])
    # format colorbar
    # format colorbar
    # divider = make_axes_locatable(ax)
    # cbar = plt.colorbar(m,shrink=0.3,pad=-0.148,ticks=None)
    # # cbar.outline.set_linewidth(0.5)
    # # for t in cbar.ax.get_yticklabels():
    # #      t.set_fontsize(fs-1.5)
    # # cbar.ax.yaxis.get_offset_text().set(size=fs-1.5)
    # cbar.set_label(colorbar_label)
    # tick_locator = ticker.MaxNLocator(nbins=6)
    # cbar.locator = tick_locator
    # cbar.update_ticks()
    
    divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", pad=0.05)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(m, shrink=0.5)
    # cbar.set_label('test label')
    set_axes_equal(ax)
    # ax.set_xlim3d([0, 4])
    ax.set_axis_off()
    # PV = (i*tstep/60*q)/total_PV
    # plt.title('PV = ' + str(PV))
    # invert z axis for matrix coordinates
    ax.invert_zaxis()
    # Set background color to white (grey is default)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    plt.show()
    
def diffusion_fun(x, t, Df, C0):
    # Equation for concentration profile as a function of space (x) and time (t)
    C = C0*(erfc((x)/(2*np.sqrt(Df*t))))
    # Return the concentration (C) from this function
    return C


def diffusion_integration_fun(dx, t, Df, C0):
    # Equation for concentration profile as a function of space (x) and time (t)
    x = np.linspace(0, dx/2, num=100) # x in meters
    M1d = np.zeros[len(t), x]
    for i in range(0, len(t)):
        for j in range(0, len(x)):
            for k in range(0, len(C0)):
                C = C0*(erfc((x)/(2*np.sqrt(Df*t[i]))))
                M1d[i, j] = trapz(C, x)
    
    return M1d
