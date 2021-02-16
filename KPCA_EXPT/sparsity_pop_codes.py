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

# get grating data
dataroot = 'grating_data'
db = np.load(os.path.join(dataroot, 'database.npy'), allow_pickle=True)
fs = []
myfont = 6
myaxis_font = 8
plt.rcParams.update({'font.size': myfont})


def optimize_l1(Psi, eta, T):
    O = np.eye(Psi.shape[0])
    mu = 500
    losses = []
    for t in range(T):
        grad = 1/Psi.shape[1] * np.sign(O @ Psi) @ Psi.T
        delta = - eta * grad + eta* O @ grad.T @ O - eta*mu * O *(O.T @ O - np.eye(O.shape[0]))
        #delta = - eta * grad - eta*mu * O *(O.T @ O - np.eye(O.shape[0]))
        O += delta
        loss = np.mean( np.abs(O @ Psi) )
        losses += [loss]
    Q, R = np.linalg.qr(O)
    print(np.mean( (Q.T @ Q - np.eye(O.shape[0]))**2 ) )
    losses[-1] = np.mean( np.abs(Q @ Psi) )
    return losses


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


    Nvals = [10,25,50,100,250,500, 1000]
    pvals = []
    KS_stats = []
    non_neg_tot = resp_avg - np.outer( np.amin(resp_avg, axis = 1), np.ones(resp_avg.shape[1]) )
    norm_tot = np.mean(np.abs(non_neg_tot))
    for i, N in enumerate(Nvals):
        num_samples = 50
        sparse = []
        true_codes = []
        for t in range(num_samples):

            randints = np.random.randint(0,resp_avg.shape[0], N)
            resp = resp_avg[randints,:]
            #resp_c = resp
            # Fix this code!!
            # r = (a - sm ) /sigma = a/sigma - sm/sigma
            #resp_c = resp - np.outer( np.amin(resp, axis = 1), np.ones(resp.shape[1]) )
            true_codes += [np.mean(np.abs(resp))]

            #resp_rot = resp - spont_shift[randints,np.newaxis]
            resp_rot = resp - resp.mean(axis=1)[:,np.newaxis]

            Q, _ = np.linalg.qr(np.random.standard_normal((N,N)), 'complete')

            code_t = Q.T @ resp
            #code_t += - np.outer( np.amin(code_t, axis = 1), np.ones(code_t.shape[1]))
            code_t += - np.amin(code_t, axis = 1)[:,np.newaxis]
            sparse += [np.mean(np.abs(code_t))]

        end = time.time()
        KS, p = sp.stats.ttest_ind(sparse, true_codes, equal_var = False)
        pvals += [p]
        KS_stats += [KS]

        print(1-KS)
        print(pvals[i])

        if i == 0 or i == len(Nvals)-1:
            plt.figure(figsize = (2,1.8))
            plt.hist(sparse, label = 'random rotation', color = 'aqua')
            #plt.plot(np.mean(np.abs(resp))*np.ones(10), np.linspace(0,1,10), '--', color = 'black')
            plt.hist(true_codes, label = 'true codes', color = 'black')
            #plt.plot(np.ones(4)*losses[-1], np.linspace(0,100,4), label = 'optimized l1')
            if i == len(Nvals)-1:
                plt.legend()
            plt.xlabel(r'Average Spike Count', fontsize=myaxis_font)
            plt.ylabel('Count', fontsize = myaxis_font)
            plt.title(r'$N = %d$' % (N) ,fontsize=myaxis_font)
            plt.tight_layout()
            plt.savefig('random_rotations_zscores_N_%d.pdf' % N)
            plt.show()
    plt.figure(figsize = (2,1.8))
    plt.loglog(Nvals, pvals, 'o', markersize=2.5)
    plt.xlabel(r'$N$', fontsize=myaxis_font)
    plt.ylabel(r'$p$-value', fontsize = myaxis_font)
    plt.title(r'$T$ Test', fontsize=myaxis_font)
    plt.tight_layout()
    plt.savefig('pvals_vs_N.pdf')
    plt.show()

    plt.figure(figsize = (1.8,1.8))
    plt.loglog(Nvals, KS_stats, 'o', markersize=1.5)
    plt.xlabel(r'$N$', fontsize=myaxis_font)
    plt.ylabel(r'$T$ Statistic', fontsize = myaxis_font)
    plt.title('T Test', fontsize=myaxis_font)
    plt.tight_layout()
    plt.savefig('T_stat_vs_N.pdf')
    plt.show()


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
