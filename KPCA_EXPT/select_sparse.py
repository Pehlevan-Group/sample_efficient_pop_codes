import numpy as np
from matplotlib import pyplot as plt
import os
import glob
from scipy.stats import zscore
import importlib
import zipfile
import math
import utils
import scipy as sp
from scipy import io
from scipy.sparse.linalg import eigsh
import csv
import power_law
import sys
import time
import scipy.stats
import jax.numpy as jnp
import jax.example_libraries.optimizers as optimizers
from jax import grad
import jax.scipy as jsp

# get grating data
dataroot = 'grating_data'
db = np.load(os.path.join(dataroot, 'database.npy'), allow_pickle=True)
fs = []
myfont = 6
myaxis_font = 8
plt.rcParams.update({'font.size': myfont})
fig_dir = 'figures/'

def optimize_l1(Psi, eta, T):
    O = np.eye(Psi.shape[0])
    mu = 500
    losses = []
    for t in range(T):
        grad = 1/Psi.shape[1] * np.sign(O @ Psi) @ Psi.T
        #delta = - eta * grad + eta* O @ grad.T @ O - eta*mu * O *(O.T @ O - np.eye(O.shape[0]))
        delta = - eta * grad - eta*mu * O *(O.T @ O - np.eye(O.shape[0]))
        O += delta
        loss = np.mean( np.abs(O @ Psi) )
        losses += [loss]
    Q, R = np.linalg.qr(O)
    print(np.mean( (Q.T @ Q - np.eye(O.shape[0]))**2 ) )
    losses[-1] = np.mean( np.abs(Q @ Psi) )
    return losses

def softmin(Z, beta=5):
    return jnp.exp(-beta * Z) / jnp.exp(-beta * Z).sum(axis=1)[:,jnp.newaxis]

def loss_fn(Q, Psi):
    Z = Q @ Psi # rotation in N dim neural space
    Zmin = softmin(Z)
    return jnp.mean(Z - Zmin)

def optimize_l1_jax(Psi, eta, T):
    Q = jnp.eye(Psi.shape[0])
    #opt_init, opt_update, get_params = optimizers.sgd(1e-3)
    #opt_state = opt_init(Q)
    g = grad(loss_fn, 0)
    for t in range(T):
        gt = g(Q, Psi)
        at= gt @ Q.T - Q @ gt.T
        Q = jsp.linalg.expm(-eta * at) @ Q
        Z = Q @ Psi
        true_cost = jnp.mean(Z - jnp.amin(Z, axis = 1)[:,jnp.newaxis])
        print( true_cost )
    return Q


all_mouse_names = []
mouse_dict = {}
for di in db:
    mname = di['mouse_name']
    if mname not in mouse_dict:
        mouse_dict[mname] = []

    datexp = di['date']
    blk = di['block']
    stype = di['expt']

    fname = '%s_%s_%s_%s.npy'%(stype, mname, datexp, blk)
    fs.append(os.path.join(dataroot, fname))

count = 0
maxcount = 5

npc = 20
fs_all = fs
all_spectra = []
all_alphas = []


count = 0
num_stim = 200

print(fs[0])
for t,f in enumerate(fs):

    if t > 0:
        break

    F1_indices = []
    F2_indices = []

    if count > 5:
        break
    count += 1

    dat = np.load(f, allow_pickle=True).item()
    # compile the responses
    #sresp, spont_shift, istim, itrain, itest = utils.compile_resp_spont_shift(dat, npc=npc, normalize=True)
    sresp, istim, itrain, itest = utils.compile_resp_spont_shift(dat, npc=npc, normalize=False)
    # trial averaging to smooth out the responses

    stim_vals = np.linspace(0,2*math.pi, num_stim)
    resp_avg = np.zeros( (sresp.shape[0], num_stim) )
    density = np.zeros( len(stim_vals))
    for i in range(num_stim-1):
        stim_inds = [j for j in range(len(istim)) if istim[j] < stim_vals[i+1] and istim[j] > stim_vals[i]]
        resp_avg[:,i] = np.mean( sresp[:,stim_inds] , axis = 1)
        density[i] = len(stim_inds)

    print(resp_avg.shape[0])
    #losses = optimize_l1( resp_avg[0:1000,:], 1e-4, 1000)
    #plt.plot(losses)
    #plt.show()

    print(resp_avg.shape[1])
    N = 1200
    r = resp_avg[0:N,:]
    R = r - np.amin(r,axis = 1)[:,np.newaxis]

    LS = np.var(R, axis = 1) / np.mean(R**2, axis=1)
    PS = np.var(R, axis = 0) / np.mean(R**2, axis = 0)


    Q,_ = np.linalg.qr(np.random.standard_normal((N,N)), 'complete')
    z = Q @ r
    R_r = z - np.amin(z, axis=1)[:,np.newaxis]

    LS_r = np.var(R_r, axis = 1) / np.mean(R_r**2, axis=1)
    PS_r = np.var(R_r, axis = 0) / np.mean(R_r**2, axis=0)

    plt.figure(figsize = (2,1.6))
    plt.hist(LS, density=True, label = 'true code', color = 'black')
    plt.hist(LS_r, density = True, label = 'RROS', color = 'aqua')
    plt.xlabel(r'Lifetime Sparseness', fontsize=myaxis_font)
    plt.ylabel(r'Density', fontsize=myaxis_font)
    #plt.title('Gratings', fontsize=myaxis_font)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir + 'lifetime_sparsenss_hist_grating.pdf')
    plt.show()

    plt.figure(figsize = (2,1.6))
    plt.hist(PS, density = True, label = 'true code', color = 'black')
    plt.hist(PS_r, density = True, label = 'RROS', color = 'aqua')
    plt.xlabel(r'Population Sparseness', fontsize=myaxis_font)
    plt.ylabel(r'Density', fontsize=myaxis_font)
    #plt.title('Gratings', fontsize=myaxis_font)

    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir + 'population_sparsenss_hist_grating.pdf')
    plt.show()

    Nvals = [10,25,50,100,250,500,1000]
    LS_N = []
    PS_N = []
    LS_r_N = []
    PS_r_N = []

    num_std_LS_mean = []
    num_std_PS_mean = []
    num_std_LS_std = []
    num_std_PS_std = []

    for i, N in enumerate(Nvals):
        ri = r[0:N,:]
        R = ri - np.amin(ri,axis = 1)[:,np.newaxis]
        print(R)
        mean_R2 = np.mean(R**2, axis = 1)
        mean_R2 = (mean_R2 > 0) * mean_R2 + (mean_R2 == 0)
        LS = np.var(R, axis = 1) / mean_R2
        mean_R2 = np.mean(R**2, axis = 0)
        mean_R2 = (mean_R2 > 0) * mean_R2 + (mean_R2 == 0)
        PS = np.var(R, axis = 0) / mean_R2

        LS_N += [LS.mean()]
        PS_N += [PS.mean()]

        num_std_LS = []
        num_std_PS = []
        for nr in range(10):
            all_LS_rot = []
            all_PS_rot = []
            for n in range(25):
                Q,_ = np.linalg.qr(np.random.standard_normal((N,N)), 'complete')
                z = Q @ ri
                R_r = z - np.amin(z, axis=1)[:,np.newaxis]

                LS_r = np.var(R_r, axis = 1) / np.mean(R_r**2, axis=1)
                PS_r = np.var(R_r, axis = 0) / np.mean(R_r**2, axis=0)

                all_LS_rot += [LS_r.mean()]
                all_PS_rot += [PS_r.mean()]

            num_std_LS += [ (LS.mean()-np.mean(all_LS_rot) ) / np.std(all_LS_rot) ]
            num_std_PS += [ (PS.mean()-np.mean(all_PS_rot) ) / np.std(all_PS_rot)  ]

        num_std_LS_mean += [np.mean(num_std_LS)]
        num_std_PS_mean += [np.mean(num_std_PS)]
        num_std_LS_std += [np.std(num_std_LS)]
        num_std_PS_std += [np.std(num_std_PS)]

    plt.figure(figsize = (2,1.6))
    plt.errorbar(Nvals, num_std_LS_mean, num_std_LS_std, fmt='o', markersize=2, color = 'black', label = 'LS')
    plt.errorbar(Nvals, num_std_PS_mean, num_std_PS_std, fmt='o', markersize=2, color = 'red', label = 'PS')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$N$', fontsize=myaxis_font)
    plt.ylabel(r'Std. Devs from Mean', fontsize=myaxis_font)
    #plt.title('Gratings', fontsize=myaxis_font)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir + 'stds_life_pop_sparsenss_hist_grating.pdf')
    plt.show()
