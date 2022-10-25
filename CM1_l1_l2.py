## The script calculates the l1 and l2 for pressure redistribution closure
## The parameters are calculated with the LVL2 simplification and 
## also directly from the pressure pressure covariance terms.

import matplotlib.pyplot as plt
import sys
import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter
from netCDF4 import Dataset
from sklearn.linear_model import LinearRegression
import pickle 
import matplotlib as mpl

sys.path.append('/glade/work/masih/Python/LES-analysis/')

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

G = 9.81
# timeidx = 10  # SBL
# for casen in ['SBL-15', 'SBL-08', 'SBL-04']:
for casen in ['CBL24-01']:
    rdir = casedir[casen]
    fname = casen
    if casen.startswith('S'):
        trange = range(26,41)
    elif casen.startswith('N'):
        trange = range(53, 68)
    elif casen.startswith('C'):
        trange = range(51, 66)
    else:
        print('undefined case')

    for SIGMA in [80, 160]:
        tke_q2 = []
        tke_adv = []
        tke_tur = []
        tke_pre = []
        tke_l1 = []
        tke_pl1 = []
        tke_C = []
        tke_pC = []
        tke_l2 = []
        tke_pl2 = []
        
        for timeidx in trange: 
            with Dataset(rdir + 'cm1out_0000%.2d.nc' % timeidx) as ds:
                # Get data
                xf = ds['xf'][:]
                U = ds['u'][0, :]
                V = ds['v'][0, :]
                W = ds['w'][0, :]
                T = ds['th'][0, :]

                rho = ds['rho'][0, :]
                m11 = ds.variables['out1'][0, :] / rho
                m22 = ds.variables['out2'][0, :] / rho
                m33 = ds.variables['out3'][0, :] / rho
                m12 = ds.variables['out4'][0, :] / rho
                m13 = ds.variables['out5'][0, :] / rho
                m23 = ds.variables['out6'][0, :] / rho

                mt1 = ds.variables['out7'][0, :] / rho
                mt2 = ds.variables['out8'][0, :] / rho
                mt3 = ds.variables['out9'][0, :] / rho

                kmh = ds['kmh'][0, :]
                kmv = ds['kmv'][0, :]
                khh = ds['khh'][0, :]
                khv = ds['khv'][0, :]

                stke = ds['tke'][0, :]

                zh = ds['zh'][:]
                zf = ds['zf'][:]            
                zi = ds['zi'][:]
                p = ds['prs'][0, :]

                ud = (U[:, :, 1:]+U[:, :, :-1])/2
                vd = (V[:, 1:, :]+V[:, :-1, :])/2
                wd = (W[1:, :, :]+W[:-1, :, :])/2

                DX = np.mean(np.diff(xf)) * 1000
                DZ = np.mean(np.diff(zf)) * 1000

            UF = uniform_filter
            arg = {'size': (0, SIGMA, SIGMA), 'mode': 'wrap'}

            uf = UF(U, **arg)
            vf = UF(V, **arg)
            wf = UF(W, **arg)
            tf = UF(T, **arg)

            up = ud - (uf[:, :, 1:] + uf[:, :, :-1])/2
            vp = vd - (vf[:, 1:, :] + vf[:, :-1, :])/2
            wp = wd - (wf[1:, :, :] + wf[:-1, :, :])/2
            tp = T - tf
            pp = p - UF(p, **arg)

            ux = np.diff(uf, axis=2) / DX
            uy = np.diff(di2(uf, axis=(0, 0, 1)), axis=1) / DX
            uz = np.diff(di2(uf, axis=(0, 0, 1)), axis=0) / DZ

            vx = np.diff(di2(vf, axis=(0, 1, 0)), axis=2) / DX
            vy = np.diff(vf, axis=1) / DX
            vz = np.diff(di2(vf, axis=(0, 1, 0)), axis=0) / DZ

            wx = np.diff(di2(wf, axis=(1, 0, 0)), axis=2) / DX
            wy = np.diff(di2(wf, axis=(1, 0, 0)), axis=1) / DX
            wz = np.diff(wf, axis=0) / DZ

            tx = np.diff(tf, axis=2) / DX
            ty = np.diff(tf, axis=1) / DX
            tz = np.diff(tf, axis=0) / DZ

            ux1 = np.diff(U, axis=2) / DX  # subgrid values
            vy1 = np.diff(V, axis=1) / DX  # subgrid values
            wz1 = np.diff(W, axis=0) / DZ  # subgrid values
            div = ux1 + vy1 + wz1

            upx = np.diff(up, axis=2) / DX
            upy = np.diff(up, axis=1) / DX
            upz = np.diff(up, axis=0) / DZ

            vpx = np.diff(vp, axis=2) / DX
            vpy = np.diff(vp, axis=1) / DX
            vpz = np.diff(vp, axis=0) / DZ

            wpx = np.diff(wp, axis=2) / DX
            wpy = np.diff(wp, axis=1) / DX
            wpz = np.diff(wp, axis=0) / DZ

            tpx = np.diff(tp, axis=2) / DX
            tpy = np.diff(tp, axis=1) / DX
            tpz = np.diff(tp, axis=0) / DZ

            u2f = uniform_filter(up * up, **arg)
            v2f = uniform_filter(vp * vp, **arg)
            w2f = uniform_filter(wp * wp, **arg)
            uvf = uniform_filter(up * vp, **arg)
            uwf = uniform_filter(up * wp, **arg)
            vwf = uniform_filter(vp * wp, **arg)

            utf = uniform_filter(up * tp, **arg)
            vtf = uniform_filter(vp * tp, **arg)
            wtf = uniform_filter(wp * tp, **arg)
            t2f = uniform_filter(tp * tp, **arg)

            t11 = u2f + UF(2/3 * di2(stke, axis=(1, 0, 0)) + 2/3
                           * di2(kmh, axis=(1, 0, 0)) * div, **arg)
            t22 = v2f + UF(2/3 * di2(stke, axis=(1, 0, 0)) + 2/3
                           * di2(kmh, axis=(1, 0, 0)) * div, **arg)
            t33 = w2f + UF(2/3 * di2(stke, axis=(1, 0, 0)) + 2/3
                           * di2(kmv, axis=(1, 0, 0)) * div, **arg)

            t12 = uvf + m12
            t13 = uwf + m13
            t23 = vwf + m23

            bx = utf + mt1
            by = vtf + mt2
            bz = wtf + mt3

            q = (t11 + t22 + t33) ** .5

            a = {}
            b = {}
            c = {}
            a[1] = di2(np.ma.array(-q / 3 * (t11 - q ** 2 / 3))).flatten()
            a[2] = di2(np.ma.array(-q / 3 * (t22 - q ** 2 / 3))).flatten()
            a[3] = di2(np.ma.array(-q / 3 * (t33 - q ** 2 / 3))).flatten()
            a[4] = di2(np.ma.array(-q / 3 * (t12))).flatten()
            a[5] = di2(np.ma.array(-q / 3 * (t13))).flatten()
            a[6] = di2(np.ma.array(-q / 3 * (t23))).flatten()

            b[1] = di2(np.ma.array(2 * q ** 2 * ux)).flatten()
            b[2] = di2(np.ma.array(2 * q ** 2 * vy)).flatten()
            b[3] = di2(np.ma.array(2 * q ** 2 * wz)).flatten()
            b[4] = np.ma.array(di2(q ** 2) * (di2(uy, axis=(1, 0, 1)) + di2(vx, axis=(1, 1, 0)))).flatten()
            b[5] = np.ma.array(di2(q ** 2) * (di2(uz, axis=(0, 1, 1)) + di2(wx, axis=(1, 1, 0)))).flatten()
            b[6] = np.ma.array(di2(q ** 2) * (di2(vz, axis=(0, 1, 1)) + di2(wy, axis=(1, 0, 1)))).flatten()

            c[1] = np.ma.array(2 * di2(pp / rho, axis=(1, 1, 1)) * di2(upx, axis=(1, 1, 0))).flatten()
            c[2] = np.ma.array(2 * di2(pp / rho, axis=(1, 1, 1)) * di2(vpy, axis=(1, 0, 1))).flatten()
            c[3] = np.ma.array(2 * di2(pp / rho, axis=(1, 1, 1)) * di2(wpz, axis=(0, 1, 1))).flatten()
            c[4] = np.ma.array(di2(pp / rho, axis=(1, 1, 1)) * (di2(upy, axis=(1, 0, 1)) + di2(vpx, axis=(1, 1, 0)))).flatten()
            c[5] = np.ma.array(di2(pp / rho, axis=(1, 1, 1)) * (di2(upz, axis=(0, 1, 1)) + di2(wpx, axis=(1, 1, 0)))).flatten()
            c[6] = np.ma.array(di2(pp / rho, axis=(1, 1, 1)) * (di2(vpz, axis=(0, 1, 1)) + di2(wpy, axis=(1, 0, 1)))).flatten()            

            shape = di2(q).shape
            l1 = np.zeros(shape[0])
            C = np.zeros(shape[0])
            for k in range(shape[0]):
                a1 = np.hstack([np.reshape(a[i], shape)[k].flatten() for i in range(1, 7)])
                b1 = np.hstack([np.reshape(b[i], shape)[k].flatten() for i in range(1, 7)])
                c1 = np.hstack([np.reshape(c[i], shape)[k].flatten() for i in range(1, 7)])

                X = np.ma.fix_invalid(a1)
                Y = np.ma.fix_invalid(b1)
                Z = np.ma.fix_invalid(c1)
                reg = LinearRegression().fit(np.array([X, Y]).T, Z)
                l1[k] = 1 / reg.coef_[0]
                C[k] = reg.coef_[1]

            tke_pl1.append(l1)
            tke_pC.append(C)

            a = {}
            b = {}
            c = {}
            qm = di2(q)
            t11m = di2(t11)
            t22m = di2(t22)
            t33m = di2(t33)
            t12m = di2(t12)
            t13m = di2(t13)
            t23m = di2(t23)

            uxm = di2(ux)
            uym = di2(uy, axis=(1, 0, 1))
            uzm = di2(uz, axis=(0, 1, 1))
            vxm = di2(vx, axis=(1, 1, 0))
            vym = di2(vy)
            vzm = di2(vz, axis=(0, 1, 1))
            wxm = di2(wx, axis=(1, 1, 0))
            wym = di2(wy, axis=(1, 0, 1))
            wzm = di2(wz)

            a[1] = di2(np.ma.array(t11 - q ** 2 / 3)).flatten()
            a[2] = di2(np.ma.array(t22 - q ** 2 / 3)).flatten()
            a[3] = di2(np.ma.array(t33 - q ** 2 / 3)).flatten()
            a[4] = di2(np.ma.array(t12)).flatten()
            a[5] = di2(np.ma.array(t13)).flatten()
            a[6] = di2(np.ma.array(t23)).flatten()

            b[1] = di2(np.ma.array(- 6 * q * ux)).flatten()
            b[2] = di2(np.ma.array(- 6 * q * vy)).flatten()
            b[3] = di2(np.ma.array(- 6 * q * wz)).flatten()
            b[4] = np.ma.array(- 3 * qm * (vxm + uym)).flatten()
            b[5] = np.ma.array(- 3 * qm * (wxm + uzm)).flatten()
            b[6] = np.ma.array(- 3 * qm * (wym + vzm)).flatten()

            temp = di2(2 * t11 * ux - t22 * vy - t33 * wz) + t12m * (2 * uym - vxm) + t13m * (2 * uzm - wxm) - t23m * (vzm + wym)
            c[1] = np.ma.array(- 2 / qm * (temp + di2(bz) * 9.81/300)).flatten()
            temp = di2(- t11 * ux + 2 * t22 * vy - t33 * wz) + t12m * (2 * vxm - uym) - t13m * (uzm + wxm) + t23m * (2 * vzm - wym)
            c[2] = np.ma.array(- 2 / qm * (temp + di2(bz) * 9.81/300)).flatten()
            temp = di2(- t11 * ux - t22 * vy + 2 * t33 * wz) - t12m * (uym + vxm) + t13m * (2 * wxm - uzm) + t23m * (2 * wym - vzm)
            c[3] = np.ma.array(- 2 / qm * (temp - 2 * di2(bz) * 9.81/300)).flatten()
            temp = t11m * vxm + t22m * uym + t12m * (uxm + vym) + t13m * vzm + t23m * uzm
            c[4] = np.ma.array(- 3 / qm * temp).flatten()
            temp = t11m * wxm + t33m * uzm + t12m * wym + t13m * (wzm + uxm) + t23m * uym
            c[5] = np.ma.array(- 3 / qm * (temp - 9.81/300 * di2(bx))).flatten()
            temp = t22m * wym + t33m * vzm + t12m * wxm + t13m * vxm + t23m * (wzm + uxm)
            c[6] = np.ma.array(- 3 / qm * (temp - 9.81/300 * di2(by))).flatten()

            shape = di2(q).shape
            l1 = np.zeros(shape[0])
            C = np.zeros(shape[0])
            for k in range(shape[0]):
                a1 = np.hstack([np.reshape(a[i], shape)[k].flatten() for i in range(1, 7)])
                b1 = np.hstack([np.reshape(b[i], shape)[k].flatten() for i in range(1, 7)])
                c1 = np.hstack([np.reshape(c[i], shape)[k].flatten() for i in range(1, 7)])

                X = np.ma.fix_invalid(a1)
                Y = np.ma.fix_invalid(b1)
                Z = np.ma.fix_invalid(c1)
                reg = LinearRegression().fit(np.array([X, Y]).T, Z)
                l1[k] = 1 / reg.coef_[0]
                C[k] = reg.coef_[1]

            tke_l1.append(l1)
            tke_C.append(C)

            a = {}
            b = {}
            c = {}
            a[1] = di2(np.ma.array(-q / 3 * bx))
            a[2] = di2(np.ma.array(-q / 3 * by))
            a[3] = di2(np.ma.array(-q / 3 * bz))
            b[1] = np.ma.array(di2(pp / rho, axis=(1, 1, 1)) * di2(tpx, axis=(1, 1, 0)))
            b[2] = np.ma.array(di2(pp / rho, axis=(1, 1, 1)) * di2(tpy, axis=(1, 0, 1)))
            b[3] = np.ma.array(di2(pp / rho, axis=(1, 1, 1)) * di2(tpz, axis=(0, 1, 1)))

            tke_pl2.append((a[1] + a[2] + a[3]) / (b[1] + b[2] + b[3]))

            a = {}
            b = {}
            c = {}
            a[1] = di2(np.ma.array(-q / 3 * bx))
            a[2] = di2(np.ma.array(-q / 3 * by))
            a[3] = di2(np.ma.array(-q / 3 * bz))
            b[1] = np.ma.array(di2(t13) * di2(tz, axis=(0, 1, 1)) + di2(bz) * di2(uz, axis=(0, 1, 1)))
            b[2] = np.ma.array(di2(t23) * di2(tz, axis=(0, 1, 1)) + di2(bz) * di2(vz, axis=(0, 1, 1)))
            b[3] = np.ma.array(di2(t33) * di2(tz, axis=(0, 1, 1)) - 9.87 / 300 * di2(t2f, axis=(1, 1, 1)))
            tke_l2num.append()
            tke_l2den.append()            
            tke_l2.append((a[1] + a[2] + a[3])/(b[1] + b[2] + b[3]))
            print(casen, timeidx)
            
        data = {}
        data['C'] = np.mean(np.stack(tke_C), axis=0)
        data['pC'] = np.mean(np.stack(tke_pC), axis=0)
        data['zh'] = zh
        data['zi'] = zi
        data['l1'] = np.mean(np.stack(tke_l1), axis=0)
        data['l2'] = np.mean(np.stack(tke_l2), axis=0)
        data['pl1'] = np.mean(np.stack(tke_pl1), axis=0)
        data['pl2'] = np.mean(np.stack(tke_pl2), axis=0)
        data['wtf'] = np.mean(wtf, axis=(1, 2))
        data['us'] = np.mean(uwf ** 2 + vwf ** 2, axis=(1, 2)) ** .5
        with open('data/l1_%.2d_%s_%d' % (timeidx, casen, SIGMA), 'wb') as pk:
            pickle.dump(data, pk)


