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
num_stim = 60

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


    Pvals = np.logspace(0.2,2.85).astype('int')
    cost_ori = []
    cost_rot = []
    N = 800
    for i,P in enumerate(Pvals):
        num_samples = 50
        original = []
        rotated = []
        for t in range(num_samples):
            randints = np.random.randint(0,resp_avg.shape[1], P)
            resp = resp_avg[0:N,randints]
            true_code_cost = np.mean( resp - np.amin(resp,axis = 1)[:,np.newaxis] )
            Q, _ = np.linalg.qr(np.random.standard_normal((N,N)), 'complete')
            code_rot = Q @ resp
            code_rot += -  np.amin(code_rot, axis = 1)[:,np.newaxis]
            original += [np.mean(true_code_cost)]
            rotated += [np.mean(code_rot)]

        cost_ori += [np.mean(original)]
        cost_rot += [np.mean(rotated)]

    plt.figure(figsize = (2,1.8))
    plt.loglog(Pvals, cost_ori, 'o', label = 'Original', markersize=1)
    plt.loglog(Pvals, cost_rot, 'o', label = 'Rotated', markersize=1)
    plt.loglog(Pvals, np.sqrt(np.log(Pvals)) / np.mean(np.sqrt(np.log(Pvals)))*np.mean(cost_rot), '--', color = 'black', label = r'$\sqrt{\ln{P}}$')
    plt.xlabel(r'$P$',fontsize = myaxis_font)
    plt.ylabel(r'Average Cost', fontsize=myaxis_font)
    plt.title('Scaling with Stimuli', fontsize=myaxis_font)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir + 'rotation_scaling_w_stimuli.pdf')
    plt.show()

    """
    Nvals = [10,25,50,100,250,500]
    pvals = []
    std_devs = []
    std_devs_opt = []
    non_neg_tot = resp_avg - np.outer( np.amin(resp_avg, axis = 1), np.ones(resp_avg.shape[1]) )
    norm_tot = np.mean(np.abs(non_neg_tot))
    for i, N in enumerate(Nvals):
        num_samples = 50
        sparse = []
        #true_code_cost =
        randints = np.random.randint(0,resp_avg.shape[0], N)
        resp = resp_avg[randints,:]
        true_code_cost = np.mean( resp - np.amin(resp,axis = 1)[:,np.newaxis] )
        opt_Q = optimize_l1_jax(resp, eta=1e-5, T=1000)
        Z = opt_Q @ resp
        opt_cost = jnp.mean(Z - jnp.amin(Z, axis = 1)[:,jnp.newaxis])
        print("true code cost: %0.4f | optimized: %0.4f" % (true_code_cost, opt_cost))
        for t in range(num_samples):

            #resp_c = resp
            # Fix this code!!
            # r = (a - sm ) /sigma = a/sigma - sm/sigma
            #resp_c = resp - np.outer( np.amin(resp, axis = 1), np.ones(resp.shape[1]) )
            #true_codes += [np.mean(np.abs(resp))]

            #resp_rot = resp - spont_shift[randints,np.newaxis]
            resp_rot = resp - resp.mean(axis=1)[:,np.newaxis]

            Q, _ = np.linalg.qr(np.random.standard_normal((N,N)), 'complete')

            code_t = Q @ resp
            #code_t += - np.outer( np.amin(code_t, axis = 1), np.ones(code_t.shape[1]))
            code_t += - np.amin(code_t, axis = 1)[:,np.newaxis]
            sparse += [np.mean(np.abs(code_t))]


        std_devs += [ ( np.mean(sparse) - true_code_cost ) / np.std(sparse) ]
        std_devs_opt += [ (np.mean(sparse) - opt_cost) / np.std(sparse) ]
        #end = time.time()
        #KS, p = sp.stats.ttest_ind(sparse, true_codes, equal_var = False)
        #pvals += [p]
        #KS_stats += [KS]

        if i == 0 or i == len(Nvals)-1:
            plt.figure(figsize = (2,1.8))
            plt.hist(sparse, label = 'random rotation', color = 'aqua')
            #plt.plot(np.mean(np.abs(resp))*np.ones(10), np.linspace(0,1,10), '--', color = 'black')
            #plt.hist(, label = 'true codes', color = 'black')
            plt.plot(np.ones(5) * true_code_cost, np.linspace(0, 5, 5), '--', color = 'black')
            plt.plot(np.ones(5)*opt_cost, np.linspace(0,5,5), '--', color = 'red')
            #plt.plot(np.ones(4)*losses[-1], np.linspace(0,100,4), label = 'optimized l1')
            if i == len(Nvals)-1:
                plt.legend()
            plt.xlabel(r'Average Spike Count', fontsize=myaxis_font)
            plt.ylabel('Count', fontsize = myaxis_font)
            plt.title(r'$N = %d$' % (N) ,fontsize=myaxis_font)
            plt.tight_layout()
            plt.savefig(fig_dir + 'random_rotations_zscores_N_%d.pdf' % N)
            plt.show()
    plt.figure(figsize = (2,1.8))
    plt.loglog(Nvals, std_devs, 'o', markersize=2.5)
    plt.loglog(Nvals, std_devs_opt, 'o', markersize=2.5, color = 'red')
    #plt.loglog(Nvals, np.sqrt(Nvals), '--', color = 'black')
    plt.xlabel(r'$N$', fontsize=myaxis_font)
    plt.ylabel(r'Std. Devs from Mean', fontsize = myaxis_font)
    #plt.title(r'$T$ Test', fontsize=myaxis_font)
    plt.tight_layout()
    plt.savefig(fig_dir + 'std_devs_vs_N.pdf')
    plt.show()

    #plt.figure(figsize = (1.8,1.8))
    #plt.loglog(Nvals, KS_stats, 'o', markersize=1.5)
    #plt.xlabel(r'$N$', fontsize=myaxis_font)
    #plt.ylabel(r'$T$ Statistic', fontsize = myaxis_font)
    #plt.title('T Test', fontsize=myaxis_font)
    #plt.tight_layout()
    #plt.savefig('T_stat_vs_N.pdf')
    #plt.show()
    """




"""
    start = time.time()
    N = 1000
    resp = resp_avg[0:N,:]
    num_samples = 100
    sparse = []
    true_codes = []
    for t in range(num_samples):
        if t % 2 == 0:
            print("done 10")
            randints = np.random.randint(0,resp_avg.shape[0], N)
            resp = resp_avg[randints,:]
            resp_c = resp
            #resp_c = resp - np.outer( np.amin(resp, axis = 1), np.ones(resp.shape[1]) )
            true_codes += [np.mean(np.abs(resp_c))]
        Q, _ = np.linalg.qr(np.random.standard_normal((N,N)), 'complete')
        code_t = Q.T @ resp
        #code_t += - np.outer( np.amin(code_t, axis = 1), np.ones(code_t.shape[1]))
        sparse += [np.mean(np.abs(code_t))]
    end = time.time()
    print("TIME: %0.5f" % (end-start))
    plt.figure(figsize=(1.8,1.5))

    plt.hist(sparse, label = 'random rotation', color = 'aqua')
    #plt.plot(np.mean(np.abs(resp))*np.ones(10), np.linspace(0,1,10), '--', color = 'black')
    plt.hist(true_codes, label = 'true codes', color = 'black')
    #plt.plot(np.ones(4)*losses[-1], np.linspace(0,100,4), label = 'optimized l1')
    plt.legend()
    plt.xlabel(r'$\ell_1$ norm', fontsize=myaxis_font)
    plt.ylabel('Count', fontsize = myaxis_font)
    plt.title('Deviation from Baseline',fontsize=myaxis_font)
    plt.tight_layout()
    plt.savefig('random_rotations_zscores_no_reshift_supp_no_enforce.pdf')
    plt.show()




    N = 1000
    resp = resp_avg[0:N,:]
    num_samples = 100
    sparse = []
    true_codes = []
    for t in range(num_samples):
        if t % 2 == 0:
            print("done 10")
            randints = np.random.randint(0,resp_avg.shape[0], N)
            resp = resp_avg[randints,:]
            resp_c = resp - np.outer( np.amin(resp, axis = 1), np.ones(resp.shape[1]) )
            true_codes += [np.mean(np.abs(resp_c))]
        Q, _ = np.linalg.qr(np.random.standard_normal((N,N)), 'complete')
        code_t = Q.T @ resp
        code_t += - np.outer( np.amin(code_t, axis = 1), np.ones(code_t.shape[1]))
        sparse += [np.mean(np.abs(code_t))]
    end = time.time()
    print("TIME: %0.5f" % (end-start))
    plt.figure(figsize=(1.8,1.5))

    plt.hist(sparse, label = 'random rotation', color = 'aqua')
    #plt.plot(np.mean(np.abs(resp))*np.ones(10), np.linspace(0,1,10), '--', color = 'black')
    plt.hist(true_codes, label = 'true codes', color = 'black')
    #plt.plot(np.ones(4)*losses[-1], np.linspace(0,100,4), label = 'optimized l1')
    plt.legend()
    plt.xlabel(r'$\ell_1$ norm', fontsize=myaxis_font)
    plt.ylabel('Count', fontsize = myaxis_font)
    plt.title('L1 for Non-Negative Codes',fontsize=myaxis_font)
    plt.tight_layout()
    plt.savefig('random_rotations_zscores_supp_enforce.pdf')
    plt.show()


    print(np.mean(np.abs(resp)))


    for i in range(12):
        plt.plot(stim_vals, resp_c[i,:])
    plt.show()

    for i in range(12):
        plt.plot(stim_vals, code_t[i,:])
    plt.show()


    sresp, istim, itrain, itest = utils.compile_resp(dat, npc=npc, normalize=False)
    start = time.time()
    N = 1000

    resp = sresp[0:N,:]
    num_samples = 100
    sparse = []
    true_codes = []
    for t in range(num_samples):
        if t % 2 == 0:
            print("done 10")
            randints = np.random.randint(0,resp_avg.shape[0], N)
            resp = resp_avg[randints,:]
            resp_c = resp
            #resp_c = resp - np.outer( np.amin(resp, axis = 1), np.ones(resp.shape[1]) )
            true_codes += [np.mean(np.abs(resp_c))]
        Q, R = np.linalg.qr(np.random.standard_normal((N,N)), 'complete')
        resp = resp - np.outer(np.mean(resp, axis = 1), np.ones(resp.shape[1]))
        code_t = Q.T @ resp
        code_t += - np.outer( np.amin(code_t, axis = 1), np.ones(code_t.shape[1]))
        sparse += [np.mean(np.abs(code_t))]
    end = time.time()
    print("TIME: %0.5f" % (end-start))

    plt.figure(figsize=(1.8,1.5))

    plt.hist(sparse, label = 'random rotation', color = 'aqua')
    #plt.plot(np.mean(np.abs(resp))*np.ones(10), np.linspace(0,1,10), '--', color = 'black')
    plt.hist(true_codes, label = 'true codes', color = 'black')
    #plt.plot(np.ones(4)*losses[-1], np.linspace(0,100,4), label = 'optimized l1')
    plt.legend()
    plt.xlabel(r'$\ell_1$ norm', fontsize=myaxis_font)
    plt.ylabel('Count', fontsize = myaxis_font)
    plt.title('Spikes No Shift', fontsize=myaxis_font)
    plt.tight_layout()
    plt.savefig('random_rotations_spike_count_no_reshift_supp.pdf')
    plt.show()


    num_samples = 100
    sparse = []
    true_codes = []
    for t in range(num_samples):
        if t % 2 == 0:
            print("done 10")
            randints = np.random.randint(0,resp_avg.shape[0], N)
            resp = resp_avg[randints,:]
            resp_c = resp
            resp_c = resp - np.outer( np.amin(resp, axis = 1), np.ones(resp.shape[1]) )
            true_codes += [np.mean(np.abs(resp_c))]
        Q, R = np.linalg.qr(np.random.standard_normal((N,N)), 'complete')
        resp = resp - np.outer(np.mean(resp, axis = 1), np.ones(resp.shape[1]))
        code_t = Q.T @ resp
        code_t += - np.outer( np.amin(code_t, axis = 1), np.ones(code_t.shape[1]))
        sparse += [np.mean(np.abs(code_t))]
    end = time.time()
    print("TIME: %0.5f" % (end-start))

    plt.figure(figsize=(1.8,1.5))

    plt.hist(sparse, label = 'random rotation', color = 'aqua')
    #plt.plot(np.mean(np.abs(resp))*np.ones(10), np.linspace(0,1,10), '--', color = 'black')
    plt.hist(true_codes, label = 'true codes', color = 'black')
    #plt.plot(np.ones(4)*losses[-1], np.linspace(0,100,4), label = 'optimized l1')
    plt.legend()
    plt.xlabel(r'$\ell_1$ norm', fontsize=myaxis_font)
    plt.ylabel('Count', fontsize = myaxis_font)
    plt.title('Spikes Rotated', fontsize=myaxis_font)
    plt.tight_layout()
    plt.savefig('random_rotations_spike_count_supp.pdf')
    plt.show()
"""
