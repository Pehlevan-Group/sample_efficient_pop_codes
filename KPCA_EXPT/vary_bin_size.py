import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
from scipy.stats import zscore
import importlib
import zipfile
import math
import utils
import scipy as sp
from scipy import io
import scipy.signal
from scipy.sparse.linalg import eigsh
import csv
import power_law
import matplotlib as mpl
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

#import decoder
myfont = 6
myaxis_font = 8
plt.rcParams.update({'font.size': myfont})
line_width = 1

dataroot = 'grating_data'
db = np.load(os.path.join(dataroot, 'database.npy'), allow_pickle=True)
fs = []
fig_dir = 'figures/'

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
#fs = [fs[0]]
#fs = fs[0:1]
all_spectra = []
all_alphas = []


count = 0

all_matern_errs = []

stim_nums = [50,100,200]
num_stim = 102
#num_stim = 50
np.random.seed(2021)

for t,f in enumerate(fs):

    if t > 0:
        break

    F1_indices = []
    F2_indices = []

    if count > 5:
        break
    count += 1

    dat = np.load(f, allow_pickle=True).item()
    sresp, istim, itrain, itest = utils.compile_resp(dat, npc=npc)

    print(sresp.shape)

    # do some trial averaging to smooth out the responses...

    all_s = []
    all_v = []
    all_stim_vals = []
    all_K = []
    for i, num_stim in enumerate(stim_nums):
        stim_vals = np.linspace(0, math.pi, num_stim)
        resp_avg = np.zeros( (sresp.shape[0], num_stim) )
        density = np.zeros( len(stim_vals))
        istim = istim % math.pi
        for i in range(num_stim-1):
            stim_inds = [j for j in range(len(istim)) if istim[j] <= stim_vals[i+1] and istim[j] > stim_vals[i]]
            resp_avg[:,i] = np.mean( sresp[:,stim_inds] , axis = 1)
            density[i] = len(stim_inds)


        resp_avg = resp_avg[:,0:resp_avg.shape[1]-1]
        stim_vals = stim_vals[0:stim_vals.shape[0]-1]

        K = 1/resp_avg.shape[0] * resp_avg.T @ resp_avg
        all_K += [K]
        #plt.imshow(K, cmap = 'coolwarm')
        #plt.show()
        s,v = np.linalg.eigh(K)
        indsort = np.argsort(s)[::-1]
        all_s += [s[indsort]]
        all_v += [v[:,indsort]]
        all_stim_vals += [stim_vals]

    plt.figure(figsize = (5,1.5))
    for i, num_stim in enumerate(stim_nums):
        plt.subplot(1,3,i+1)
        plt.title('bins = %d' % num_stim)
        plt.imshow(all_K[i], cmap = 'coolwarm')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(r'$\theta_1$',fontsize=myaxis_font)
        plt.ylabel(r'$\theta_2$',fontsize=myaxis_font)
    plt.savefig(fig_dir+'bins_K_imshow.pdf')
    plt.show()

    plt.figure(figsize=(1.8,1.5))
    for i, num_stim in enumerate(stim_nums):
        plt.loglog(all_s[i][:-2]/all_s[i][0], label = 'bins = %d' % num_stim)
    plt.legend()
    plt.xlabel(r'$k$',fontsize=myaxis_font)
    plt.ylabel(r'$\lambda_k$',fontsize=myaxis_font)
    plt.tight_layout()
    plt.savefig(fig_dir+'bins_spectra.pdf')
    plt.show()

    plt.figure(figsize=(6,2))
    for i, num_stim in enumerate(stim_nums):
        plt.subplot(1,3,i+1)
        plt.title('bins = %d' % num_stim)
        for j in range(6):
            if j == 0:
                plt.ylabel(r'$\psi_k(\theta)$',fontsize=myaxis_font)
            plt.xlabel(r'$\theta$',fontsize=myaxis_font)
            plt.plot(all_stim_vals[i], all_v[i][:,j] + j*0.5)
    plt.tight_layout()
    plt.savefig(fig_dir+'bins_eigenfunctions.pdf')
    plt.show()
