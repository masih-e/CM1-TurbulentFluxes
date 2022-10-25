import matplotlib.pyplot as plt
from datetime import datetime
import os, sys
import numpy as np
import xarray as xr
from skimage.measure import block_reduce
from scipy.ndimage import gaussian_filter, uniform_filter
from netCDF4 import Dataset
from sklearn.linear_model import LinearRegression
import matplotlib.ticker as tck

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

fig, axs = plt.subplots(2, 2, figsize=(6.5, 6), sharex=True, gridspec_kw={'wspace': .4, 'hspace': .2})
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

for idx, case in enumerate(['CBL24-01', 'CBL05-15', 'NBL-15', 'SBL-08']):
    if case.startswith('C'):
        timeidx = 65
        slist = [10, 20, 40, 80, 160]
    elif case.startswith('N'):
        timeidx = 67
        slist = [10, 20, 40, 80, 160]
    else:
        timeidx = 40
        slist = [10, 20, 40, 80, 160]        

    for SIGMA in slist:
        with open('../data/bud_%.2d_%s_%d' % (timeidx, case, SIGMA), 'rb') as pk:
            data = pickle.load(pk)


        rdir = casedir[case]
        with Dataset(rdir + 'cm1out_0000%.2d.nc' % timeidx) as ds:
            diss = ds.variables['dissten'][0,:]
            xf = ds['xf'][:]

            if SIGMA == 0:
                DX = np.mean(np.diff(xf)) * 1000 * xf.shape[0] / np.mean(data['zi'])
            else:
                DX = np.mean(np.diff(xf)) * 1000 * SIGMA / np.mean(data['zi'])

        height1 = data['zh'] / np.mean(data['zi'])*1000
        height = (height1[1:]+height1[:-1])/2

        xi = np.where(data['zh']>np.mean(data['zi'])/1000)[0][0] //2
        kz = np.mean(data['zi'])/2 * .4

        X = di2(data['tke_BP'], axis=(1, 0, 0)) + data['tke_SP'] + data['tke_adv'] + (data['tke_tur'] + data['tke_pre']) + di2(diss[1:])  # residual term (diss.)
        q2 = di2(data['tke_q2'])
        lam = np.mean(q2 ** (3/2), axis=(1, 2)) / np.mean(X, axis=(1, 2))        
        s = np.std(q2 ** (3/2), axis=(1, 2)) / np.mean(X, axis=(1, 2))
        
        lt = data['lt']
        
        if SIGMA == 10:
            axs[0,0].plot(DX, lam[xi]/(np.mean(lt)*1000*.1), 'o', color=cycle[idx], label=case)
        else:
            axs[0,0].plot(DX, lam[xi]/(np.mean(lt)*1000*.1), 'o', color=cycle[idx])
        axs[0,0].axhline(y=16.6, linestyle=':', color='b')

#        X = -(data['th2_adv']+data['th2_tur']+data['th2_pro'])
        X = -(data['th2_pro']/2)
        lah = (-np.mean(data['th2_qt2'], axis=(1, 2))/np.mean(X, axis=(1, 2)))        
        axs[0,1].plot(DX, lah[xi]/(np.mean(data['lt'])*1000*.1), 'o', color=cycle[idx])
        axs[0,1].axhline(y=10.1, linestyle=':', color='b')
        
        with open('../data/l1_%.2d_%s_%d' % (timeidx, case, SIGMA), 'rb') as pk:
            data = pickle.load(pk)

        axs[1,0].plot(DX, data['l1'][xi]/(np.mean(lt)*1000*.1), 'o', color=cycle[idx])
        axs[1,0].axhline(y=0.92, linestyle=':', color='b')
        
        x = data['l2']
        m = np.mean(x, axis=(1, 2))
        s = np.std(x, axis=(1, 2))
        for k in range(x.shape[0]):
            x[k, np.logical_or(x[k]>m[k]+s[k]*4, x[k]<m[k]-s[k]*4)]=np.nan        
        
        axs[1,1].plot(DX, np.nanmean(x, axis=(1, 2))[xi]/(np.mean(lt)*1000*.1), 'o', color=cycle[idx])
        axs[1,1].axhline(y=0.74, linestyle=':', color='b')

axs[0,0].legend()
axs[0,0].set_ylabel('$\Lambda_1/L_T$')
axs[0,1].set_ylabel('$\Lambda_2/L_T$')
axs[1,0].set_ylabel('$l_1/L_T$')
axs[1,1].set_ylabel('$l_2/L_T$')

axs[0, 0].set_title('a) TKE diss. length')
axs[0, 1].set_title(r'b) <$\theta^2$> diss. length')
axs[1, 0].set_title('c) momentum redist. length')
axs[1, 1].set_title('d) temperature redist. length')

axs[0,1].set_ylim([0, 21])

axs[1,0].set_xlabel('$\Delta_H^*$')
axs[1,1].set_xlabel('$\Delta_H^*$')
axs[0,1].xaxis.set_major_locator(tck.MultipleLocator(base=1.0))
axs[0,1].xaxis.set_major_locator(tck.MultipleLocator(base=1.0))
axs[1,0].xaxis.set_major_locator(tck.MultipleLocator(base=1.0))
axs[1,1].xaxis.set_major_locator(tck.MultipleLocator(base=1.0))
plt.tight_layout()
plt.savefig('FIG_SUM')