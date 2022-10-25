## The script calculates the tke and t2 budgets. 
## The budget terms are used for calculating lambda 1 and 2 (diss. length)
import matplotlib.pyplot as plt
import sys
import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter
from netCDF4 import Dataset
import pickle
sys.path.append('/glade/work/masih/Python/LES-analysis/')


def di2(x, axis=(1, 1, 1)):
    if axis[0] == 1:
        x = (x[1:, :, :] + x[:-1, :, :])/2
    if axis[1] == 1:
        x = (x[:, 1:, :] + x[:, :-1, :])/2
    if axis[2] == 1:
        x = (x[:, :, 1:] + x[:, :, :-1])/2
    return x

def meanr(x):
    new = np.mean(x, axis=(1, 2)).repeat(x.shape[1]).repeat(x.shape[2]).reshape(x.shape)
    return new

casedir = {'CBL24-01': '/glade/scratch/masih/cm1/cm1r21.0_diag_wind01-2/',
           'CBL24-08': '/glade/scratch/masih/cm1/cm1r21.0_diag_wind08-2/',
           'CBL24-15': '/glade/scratch/masih/cm1/cm1r21.0_diag_wind15-2/',
           'CBL05-15': '/glade/scratch/masih/cm1/cm1r21.0_diag_wind15_wtlow-2/',
           'NBL-08': '/glade/scratch/masih/cm1/cm1r21.0_diag_neutral_8/',
           'NBL-15': '/glade/scratch/masih/cm1/cm1r21.0_diag_neutral-15-2/',
           'SBL-04': '/glade/scratch/masih/cm1/cm1r21.0_diag_SBL/les_SBL_04/',
           'SBL-08': '/glade/scratch/masih/cm1/cm1r21.0_diag_SBL/les_SBL_08/',
           'SBL-08b': '/glade/scratch/masih/cm1/cm1r21.0_diag_SBL/les_SBL_08-big/',           
           'SBL-15': '/glade/scratch/masih/cm1/cm1r21.0_diag_SBL/les_SBL_15/'}

G = 9.81

#for casen in ['CBL24-01', 'CBL24-08','CBL24-15', 'CBL05-15', 'NBL-15', 'SBL-15', 'SBL-08']:
for casen in ['NBL-15']:#, 'SBL-04', 'SBL-08', 'SBL-15']:
    rdir = casedir[casen]
    fname = casen
    for SIGMA in [10, 40, 80, 160]:
        # tke* for tke budget
        tke_q2 = []
        tke_SP = []
        tke_BP = []
        tke_adv = []
        tke_tur = []
        tke_pre = []
        tke_dis = []
        # th2* for temperature budget
        th2_qt2 = []
        th2_adv = []
        th2_tur = []
        th2_pro = []
        lt = []
        nm = []
        
        if casen.startswith('S'):
            trange = range(26,41)
        elif casen.startswith('N'):
            trange = range(53, 68)
        elif casen.startswith('C'):
            trange = range(51, 66)
        else:
            print('undefined case')

        for timeidx in trange: # SBL
            with Dataset(rdir + 'cm1out_0000%.2d.nc' % timeidx) as ds:
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

                nm.append(ds['nm'][0, :])
                kmh = ds['kmh'][0, :]
                kmv = ds['kmv'][0, :]
                khh = ds['khh'][0, :]
                khv = ds['khv'][0, :]
                stke = ds['tke'][0, :]

                zh = ds['zh'][:]
                zf = ds['zf'][:]
                x = np.argmax(np.diff(T, axis=0).mean(axis=(1, 2)))
                zi = ds['zi'][:]
                p = ds['prs'][0, :]

                ud = (U[:, :, 1:] + U[:, :, :-1])/2
                vd = (V[:, 1:, :] + V[:, :-1, :])/2
                wd = (W[1:, :, :] + W[:-1, :, :])/2

                DX = np.mean(np.diff(xf)) * 1000
                DZ = np.mean(np.diff(zf)) * 1000

            UF = uniform_filter
            arg = {'size': (0, SIGMA, SIGMA), 'mode': 'wrap'}

            if SIGMA == 0:
                uf = meanr(U)
                vf = meanr(V)
                wf = meanr(W)
                tf = meanr(T)
                pp = p - meanr(p)
            else:
                uf = UF(U, **arg)
                vf = UF(V, **arg)
                wf = UF(W, **arg)
                tf = UF(T, **arg)
                pp = p - UF(p, **arg)
            up = ud - (uf[:, :, 1:] + uf[:, :, :-1])/2
            vp = vd - (vf[:, 1:, :] + vf[:, :-1, :])/2
            wp = wd - (wf[1:, :, :] + wf[:-1, :, :])/2
            tp = T - tf



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

            tx1 = np.diff(T, axis=2) / DX  # subgrid values
            ty1 = np.diff(T, axis=1) / DX  # subgrid values
            tz1 = np.diff(T, axis=0) / DZ  # subgrid values

            if SIGMA == 0:        
                u2f = meanr(up * up)
                v2f = meanr(vp * vp)
                w2f = meanr(wp * wp)
                uvf = meanr(up * vp)
                uwf = meanr(up * wp)
                vwf = meanr(vp * wp)
                utf = meanr(up * tp)
                vtf = meanr(vp * tp)
                wtf = meanr(wp * tp)
                t2f = meanr(tp * tp)
            else:
                u2f = UF(up * up, **arg)
                v2f = UF(vp * vp, **arg)
                w2f = UF(wp * wp, **arg)
                uvf = UF(up * vp, **arg)
                uwf = UF(up * wp, **arg)
                vwf = UF(vp * wp, **arg)
                utf = UF(up * tp, **arg)
                vtf = UF(vp * tp, **arg)
                wtf = UF(wp * tp, **arg)
                t2f = UF(tp * tp, **arg)

            t11 = u2f + UF(2/3 * di2(stke, axis=(1, 0, 0)) + 2/3 *
                           di2(kmh, axis=(1, 0, 0)) * div, **arg)
            t22 = v2f + UF(2/3 * di2(stke, axis=(1, 0, 0)) + 2/3 *
                           di2(kmh, axis=(1, 0, 0)) * div, **arg)
            t33 = w2f + UF(2/3 * di2(stke, axis=(1, 0, 0)) + 2/3 *
                           di2(kmv, axis=(1, 0, 0)) * div, **arg)

            t12 = uvf + m12
            t13 = uwf + m13
            t23 = vwf + m23
            bx = utf + mt1
            by = vtf + mt2
            bz = wtf + mt3

            # TKE
            tke_q2.append(np.where((t11 + t22 + t33) > 0, (t11 + t22 + t33), 0))
            E = u2f + v2f + w2f + 2 * di2(stke, axis=(1, 0, 0))

            dz = np.diff(zf)
            den = 0
            num = 0
            for k in range(zh.shape[0]):
                num = num + zh[k] * np.sqrt(E[k]) * dz[k]
                den = den + np.sqrt(E[k]) * dz[k]
            lt.append(num/den)

            ex = di2(di2(uf, axis=(0, 0, 1)), axis=(1, 1, 1)) * \
                di2(np.diff(E, axis=2), axis=(1, 1, 0)) / DX
            ey = di2(di2(vf, axis=(0, 1, 0)), axis=(1, 1, 1)) * \
                di2(np.diff(E, axis=1), axis=(1, 0, 1)) / DX
            ez = di2(di2(wf, axis=(1, 0, 0)), axis=(1, 1, 1)) * \
                di2(np.diff(E, axis=0), axis=(0, 1, 1)) / DZ
            tke_adv.append(-(ex+ey+ez))

            T2U = UF(up * (up * up + vp * vp + wp * wp), **arg)
            T2V = UF(vp * (up * up + vp * vp + wp * wp), **arg)
            T2W = UF(wp * (up * up + vp * vp + wp * wp), **arg)
            T2UX = - di2(np.diff(T2U, axis=2) / DX, axis=(1, 1, 0))
            T2VY = - di2(np.diff(T2V, axis=1) / DX, axis=(1, 0, 1))
            T2WZ = - di2(np.diff(T2W, axis=0) / DZ, axis=(0, 1, 1))
            tke_tur.append(T2UX + T2VY + T2WZ)

            P1 = - 2 * di2(np.diff(up * pp / rho, axis=2) / DX, axis=(1, 1, 0))
            P2 = - 2 * di2(np.diff(vp * pp / rho, axis=1) / DX, axis=(1, 0, 1))
            P3 = - 2 * di2(np.diff(wp * pp / rho, axis=0) / DZ, axis=(0, 1, 1))

            tke_pre.append(UF(P1 + P2 + P3, **arg))

            tke_SP.append(- 2 * (di2(t11 * ux + t22 * vy + t33 * wz, axis=(1,1,1)) +
                            di2(t12, axis=(1,1,1)) * (di2(uy, axis=(1,0,1)) + di2(vx, axis=(1,1,0))) + \
                            di2(t13, axis=(1,1,1)) * (di2(uz, axis=(0,1,1)) + di2(wx, axis=(1,1,0))) + \
                            di2(t23, axis=(1,1,1)) * (di2(vz, axis=(0,1,1)) + di2(wy,axis=(1,0,1)))))

            tke_BP.append(2 * di2(bz, axis=(0, 1, 1)) * G / 300)

            uxm = di2(ux)
            uym = di2(uy, axis=(1, 0, 1))
            uzm = di2(uz, axis=(0, 1, 1))
            vxm = di2(vx, axis=(1, 1, 0))
            vym = di2(vy)
            vzm = di2(vz, axis=(0, 1, 1))
            wxm = di2(wx, axis=(1, 1, 0))
            wym = di2(wy, axis=(1, 0, 1))
            wzm = di2(wz)

            temp = uxm * uxm + uym * uym + uzm * uzm
            temp = temp + vxm * vxm + vym * vym + vzm * vzm
            temp = temp + wxm * wxm + wym * wym + wzm * wzm
            tke_dis.append(temp * di2(kmh[1:]+kmh[:-1])/2)

            # temperature budget
            th2_qt2.append(di2(t2f * (E ** .5), axis=(1, 1, 1)))
            Xx = di2(np.diff(t2f, axis=2), axis=(1, 1, 0))
            Xy = di2(np.diff(t2f, axis=1), axis=(1, 0, 1))
            Xz = di2(np.diff(t2f, axis=0), axis=(0, 1, 1))
            th2_adv.append(-(di2(di2(uf, axis=(0, 0, 1)), axis=(1, 1, 1)) * Xx +
                        di2(di2(vf, axis=(0, 1, 0)), axis=(1, 1, 1)) * Xy +
                        di2(di2(wf, axis=(1, 0, 0)), axis=(1, 1, 1)) * Xz))
            X2UX = di2(np.diff(UF(up * tp * tp, **arg), axis=2), axis=(1, 1, 0))
            X2VY = di2(np.diff(UF(vp * tp * tp, **arg), axis=1), axis=(1, 0, 1))
            X2WZ = di2(np.diff(UF(wp * tp * tp, **arg), axis=0), axis=(0, 1, 1))
            th2_tur.append(-(X2UX + X2VY + X2WZ))
            th2_pro.append(-2 * (di2(bx, axis=(1, 1, 1)) * di2(tx, axis=(1, 1, 0)) +
                            di2(by, axis=(1, 1, 1)) * di2(ty, axis=(1, 0, 1)) +
                            di2(bz, axis=(1, 1, 1)) * di2(tz, axis=(0, 1, 1))))

        # tke budget
        tke_q2 = np.stack(tke_q2)
        tke_SP = np.stack(tke_SP)
        tke_BP = np.stack(tke_BP)
        tke_adv = np.stack(tke_adv)
        tke_tur = np.stack(tke_tur)
        tke_pre = np.stack(tke_pre)
        tke_dis = np.stack(tke_dis)
        lt = np.stack(lt)
        nm = np.stack(nm)        
        # temp. budget
        th2_qt2 = np.stack(th2_qt2)
        th2_adv = np.stack(th2_adv)
        th2_tur = np.stack(th2_tur)
        th2_pro = np.stack(th2_pro)

        data = {}
        data['zh'] = zh
        data['zf'] = zf
        data['zi'] = zi
        data['lt'] = np.mean(lt, axis=0)
        data['nm'] = np.mean(nm, axis=0)
        data['tke_q2'] = np.mean(tke_q2, axis=0)
        data['tke_BP'] = np.mean(tke_BP, axis=0)
        data['tke_SP'] = np.mean(tke_SP, axis=0)
        data['tke_adv'] = np.mean(tke_adv, axis=0)
        data['tke_tur'] = np.mean(tke_tur, axis=0)
        data['tke_pre'] = np.mean(tke_pre, axis=0)
        data['tke_dis'] = np.mean(tke_dis, axis=0)
        data['th2_qt2'] = np.mean(th2_qt2, axis=0)
        data['th2_adv'] = np.mean(th2_adv, axis=0)
        data['th2_tur'] = np.mean(th2_tur, axis=0)
        data['th2_pro'] = np.mean(th2_pro, axis=0)
        data['wtf'] = np.mean(wtf, axis=(1, 2))
        data['us'] = np.mean(uwf ** 2 + vwf ** 2, axis=(1, 2)) ** .5
        x = np.argmax(np.diff(T, axis=0).mean(axis=(1, 2)))
        with open('data/bud_%.2d_%s_%d' % (timeidx, casen, SIGMA), 'wb') as pk:
            pickle.dump(data, pk)
