import matplotlib.pyplot as plt
from datetime import datetime
import os, sys
import numpy as np
import xarray as xr
from skimage.measure import block_reduce
from scipy.ndimage import gaussian_filter, uniform_filter
from netCDF4 import Dataset
from sklearn.linear_model import LinearRegression

sys.path.append('/glade/work/masih/Python/LES-analysis/')
import pickle 
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [10,10]
mpl.rcParams['figure.titlesize'] = 11
mpl.rcParams['figure.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['lines.linewidth'] = 1.8
mpl.rcParams['grid.linewidth'] = .25
mpl.rcParams['figure.subplot.wspace'] = 0.05
mpl.rcParams['figure.subplot.hspace'] = 0.05
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['legend.framealpha'] = .75
mpl.rcParams['legend.loc'] = 'best'
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.dpi'] = 200

import cdiff
import time
t_start = time.time()
import imp
imp.reload(cdiff)


def di2(x, axis=(1, 1, 1)):
    if axis[0] == 1:
        x = (x[1:, :, :] + x[:-1, :, :])/2
    if axis[1] == 1:
        x = (x[:, 1:, :] + x[:, :-1, :])/2
    if axis[2] == 1:
        x = (x[:, :, 1:] + x[:, :, :-1])/2
    return x

casedir = {'CBL24-01': '/glade/scratch/masih/cm1/cm1r21.0_diag_wind01-2/',
           'CBL24-08': '/glade/scratch/masih/cm1/cm1r21.0_diag_wind08-2/',
           'CBL24-15': '/glade/scratch/masih/cm1/cm1r21.0_diag_wind15-2/',
           'CBL05-15': '/glade/scratch/masih/cm1/cm1r21.0_diag_wind15_wtlow-2/',
           'NBL-08': '/glade/scratch/masih/cm1/cm1r21.0_diag_neutral_8/',
           'NBL-15': '/glade/scratch/masih/cm1/cm1r21.0_diag_neutral-15-2/',
           'SBL-04': '/glade/scratch/masih/cm1/cm1r21.0_diag_SBL/les_SBL_04/',
           'SBL-08': '/glade/scratch/masih/cm1/cm1r21.0_diag_SBL/les_SBL_08/',
           'SBL-15': '/glade/scratch/masih/cm1/cm1r21.0_diag_SBL/les_SBL_15/'}


fig, axs = plt.subplots(2, 2, figsize=(6.5, 6), gridspec_kw={'wspace': .4, 'hspace': .4})
for idx, case in enumerate(['CBL24-01', 'CBL05-15', 'NBL-15', 'SBL-08']):
    if case.startswith('C'):
        timeidx = 65
        DX = 25
    elif case.startswith('N'):
        timeidx = 67
        DX = 16
    else:
        timeidx = 40
        DX = 3.125
        
    sfc = 2
#    for SIGMA in [10, 20, 40, 80, 0]:
    for SIGMA in [10, 20, 40, 80, 160]:
        with open('../data/l1_%.2d_%s_%d' % (timeidx, case, SIGMA), 'rb') as pk:
            data = pickle.load(pk)
            
        height1 = data['zh'] / np.mean(data['zi'])*1000
        height = (height1[1:]+height1[:-1])/2
        
        x = data['l2']
        m = np.mean(x, axis=(1, 2))
        s = np.std(x, axis=(1, 2))
        for k in range(x.shape[0]):
            x[k, np.logical_or(x[k]>m[k]+s[k]*4, x[k]<m[k]-s[k]*4)]=np.nan        

        axs[idx//2, idx % 2].plot(np.nanmean(x, axis=(1, 2))[sfc:], height[sfc:], label='%2.2f' % (DX * SIGMA / np.mean(data['zi'])))
        axs[idx//2, idx % 2].set_ylim([0, 1.3])
    

axs[0, 0].set_title('a) 1 m $s^{-1}$; 0.24 K m $s^{-1}$')
axs[0, 1].set_title('b) 15 m $s^{-1}$; 0.05 K m $s^{-1}$')
axs[1, 0].set_title('c) 15 m $s^{-1}$; Neutral')
axs[1, 1].set_title('d) 8 m $s^{-1}$; Stable')

axs[0, 0].set_xlim([-1, 500])
axs[0, 1].set_xlim([-1, 500])
axs[1, 0].set_xlim([-1, 200])
axs[1, 1].set_xlim([-1, 100])

axs[0, 0].legend(title=r'$\Delta_H^*$')
axs[0, 1].legend(title=r'$\Delta_H^*$')
axs[1, 0].legend(title=r'$\Delta_H^*$')
axs[1, 1].legend(title=r'$\Delta_H^*$')

axs[0, 0].set_ylabel('Z/$z_i$')
axs[1, 0].set_ylabel('Z/$z_i$')
axs[1, 0].set_xlabel('$l_2$ (m)')
axs[1, 1].set_xlabel('$l_2$ (m)')
plt.tight_layout()
plt.savefig('FIG_l2')
