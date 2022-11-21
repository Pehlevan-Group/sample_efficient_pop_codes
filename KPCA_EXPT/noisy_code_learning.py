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


fig_dir = 'figures/'
"""
root = 'natural_images/'
f = root + 'natimg2800_M170714_MP032_2017-09-14.mat'
f2 = root + 'stimuli_class_assignment_confident.mat'
f3 = root + 'images_natimg2800_all.mat'
dat = sp.io.loadmat(f)

f3 = sp.io.loadmat(f3)
imgs = f3['imgs']


# classes maps stim id to a class

m2 = sp.io.loadmat(f2)
classes = m2['class_assignment'][0]
class_names = m2['class_names']
print(class_names)
print(classes)

print("classes shape")
print(classes.shape)


resp = dat['stim'][0]['resp'][0] # stim x neurons
spont = dat['stim'][0]['spont'][0] # timepts x neurons
istim = (dat['stim'][0]['istim'][0]).astype(np.int32) # stim ids
istim -= 1 # get out of MATLAB convention

print("neural response shape")
print(resp.shape)
# stim is preprocessed
istim = istim[:,0]
nimg = istim.max() # these are blank stims (exclude them)
resp = resp[istim<nimg, :]
istim = istim[istim<nimg]
print(np.amax(istim))
print("istim.shape")
print(istim.shape)

# subtract spont (32D)
mu = spont.mean(axis=0)
sd = spont.std(axis=0) + 1e-6
resp = (resp - mu) / sd
spont = (spont - mu) / sd
#sv,u = eigsh(spont.T @ spont, k=32)
#resp = resp - (resp @ u) @ u.T

# mean center each neuron
resp -= resp.mean(axis=0)
resp = resp / (resp.std(axis = 0)+1e-6)



# get classes
print("istim")
print(istim)
class_stim = classes[istim]
print("class stim shape")
print(class_stim.shape)
print(class_stim)


# which experimental trials belong to which category
inds1 = [i for i in range(len(class_stim)) if class_stim[i] == 1]
inds2 = [i for i in range(len(class_stim)) if class_stim[i] == 7]
inds_12 = inds1 + inds2

# which images belong to which category
imgs_inds1 = [i for i in range(len(classes)) if classes[i] == 1]
imgs_inds2 = [i for i in range(len(classes)) if classes[i] == 7]


print("imgs shape")
print(imgs.shape)
np.random.seed(0)
A = imgs[:, 90:180, imgs_inds1]
B = imgs[:, 90:180, imgs_inds2]


y = class_stim[inds_12]
a = np.amin(y)
b = np.amax(y)

y = 2/(b-a)*(y-np.mean([a,b]))


def sorted_spectral_decomp(resp, imgs=None):
    K = 1/resp.shape[1] * resp @ resp.T
    #inds_0 = [i for i in range(len(class_stim)) if class_stim[i] == 0 ]
    #inds_1 = [i for i in range(len(class_stim)) if class_stim[i] == 1 ]
    #inds_sort = inds_0 + inds_1
    #k = K[inds_sort,:]
    plt.imshow(K)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(r'image 1', fontsize=20)
    plt.ylabel(r'image 2', fontsize=20)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(fig_dir+ 'kernel_matrix_natural_images.pdf')
    plt.show()
    print(K.shape)
    s,v = np.linalg.eigh(K)
    print(s.shape)
    indsort = np.argsort(s)[::-1]
    s = s[indsort]
    v = v[:,indsort]
    print(s.shape)
    return K,s,v

resp_12 = resp[inds_12,:]

K, s, v = sorted_spectral_decomp(resp_12, imgs)
#coeffs, s, theory_lc, err_expt = compute_learning_curves(K, y, pvals, lamb)

num_repeats=10
lamb = 1.0
pvals = np.logspace(0.5, 3.2).astype('int')
err_expt = np.zeros((len(pvals),num_repeats))
for n in range(num_repeats):
    for i,p in enumerate(pvals):
        permed = np.random.permutation(K.shape[0])
        rand_i = permed[0:p]
        test_i = permed[p:permed.shape[0]]
        Ki = K[rand_i,:]
        Kii = Ki[:,rand_i]
        K_test = Ki[:,test_i]
        print(Ki.shape)
        yi = y[rand_i]
        yhat = K_test.T @ np.linalg.inv(Kii + 1/K.shape[0]*lamb*np.eye(p)) @ yi
        err_expt[i,n] = np.sum( np.mean( (y[test_i] - yhat )**2 , axis = 0 ) )

plt.figure(figsize=(1.8,1.5))
plt.semilogx(pvals, np.mean(err_expt, axis = 1))
plt.xlabel(r'$P$',fontsize=myaxis_font)
plt.ylabel(r'$E_g$',fontsize=myaxis_font)
plt.savefig(fig_dir+ 'noisy_code_natural_img.pdf')
plt.show()

"""




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


npc = 0
fs_all = fs
#fs = [fs[0]]
#fs = fs[0:1]
all_spectra = []
all_alphas = []


count = 0

all_matern_errs = []

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
    print(istim)
    print("min/max istim")
    print(np.amin(istim))
    print(np.amax(istim))

    K = 1/sresp.shape[0] * sresp.T @ sresp

    s,v = np.linalg.eigh(K)
    K = K/np.amax(s)

    y1_tot = np.sign(np.cos(2*istim))
    y2_tot = np.sign(np.cos(6*istim))

    num_repeats = 10
    pvals = np.linspace(2, 1000, 20).astype('int')
    #pvals = np.logspace(0.4,2, 12).astype('int')
    #num_repeats = 30
    lamb = 0.0001

    err_easy = np.zeros((len(pvals), num_repeats))
    err_hard = np.zeros((len(pvals), num_repeats))

    ### need to do test train split. Otherwise memorization of noise is possible

    for n in range(num_repeats):
        for i,p in enumerate(pvals):
            rand_i = np.random.randint(0,K.shape[0],p)
            Ki = K[rand_i,:]
            Kii = Ki[:,rand_i]
            x = istim[rand_i]
            y1 = np.sign(np.cos(2*x))
            y2 = np.sign(np.cos(6*x))
            #y1 = u[rand_i,0]
            #y2 = u[rand_i,5]
            yhat1 = Ki.T @ np.linalg.inv(Kii + 1/K.shape[0]*lamb*np.eye(p)) @ y1
            yhat2 = Ki.T @ np.linalg.inv(Kii + 1/K.shape[0]*lamb*np.eye(p)) @ y2

            yhat1_te = yhat1[~rand_i]
            yhat2_te = yhat2[~rand_i]

            err_easy[i,n] = np.mean( (y1_tot[~rand_i] - yhat1_te )**2 )
            err_hard[i,n] = np.mean( (y2_tot[~rand_i] - yhat2_te )**2 )



    num_stim = 500
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

    K_avg = 1/resp_avg.shape[0] * resp_avg.T @ resp_avg
    s, v=np.linalg.eigh(K_avg)
    K_avg = K_avg/np.amax(s)
    #pvals_sub = np.linspace(10,500, 12).astype('int')
    num_repeats = 30
    lamb = 1.0
    y_easy_true = np.sign(np.cos(2*stim_vals))
    y_hard_true = np.sign(np.cos(6*stim_vals))
    err_easy_sub = np.zeros((len(pvals), num_repeats))
    err_hard_sub = np.zeros((len(pvals), num_repeats))
    for n in range(num_repeats):
        for i,p in enumerate(pvals):
            rand_i = np.random.randint(0,K_avg.shape[0],p)
            Ki = K_avg[rand_i,:]
            Kii = Ki[:,rand_i]
            x = stim_vals[rand_i]
            y1 = np.sign(np.cos(2*x))
            y2 = np.sign(np.cos(6*x))
            #y1 = u[rand_i,0]
            #y2 = u[rand_i,5]
            yhat1 = Ki.T @ np.linalg.inv(Kii + 1/K.shape[0]*lamb*np.eye(p)) @ y1
            yhat2 = Ki.T @ np.linalg.inv(Kii + 1/K.shape[0]*lamb*np.eye(p)) @ y2
            err_easy_sub[i,n] = np.mean( (y_easy_true - yhat1 )**2 )
            err_hard_sub[i,n] = np.mean( (y_hard_true - yhat2 )**2 )



    plt.figure(figsize=(1.8,1.5))
    plt.plot(pvals, np.mean(err_easy, axis = 1), color = 'blue', label = 'Low Freq.')
    plt.plot(pvals, np.mean(err_hard, axis = 1), color = 'red',  label = 'High Freq.')
    plt.plot(pvals, np.mean(err_easy_sub, axis = 1), '--',color = 'blue', label = 'High Freq. Avg')
    plt.plot(pvals, np.mean(err_hard_sub, axis = 1), '--',color = 'red', label = 'Low Freq. Avg')
    plt.xlabel(r'$P$',fontsize=myaxis_font)
    plt.ylabel(r'$E_g$',fontsize=myaxis_font)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir + 'noisy_code_learning_curves.pdf')
    plt.show()

    p = 100
    lamb_vals = np.logspace(-5,1.5,10)
    err_lamb_easy = np.zeros((len(lamb_vals), num_repeats))
    err_lamb_hard = np.zeros((len(lamb_vals), num_repeats))
    err_lamb_easy_avg = np.zeros((len(lamb_vals), num_repeats))
    err_lamb_hard_avg = np.zeros((len(lamb_vals), num_repeats))
    for n in range(num_repeats):
        for i, lamb in enumerate(lamb_vals):
            rand_i = np.random.randint(0,K.shape[0],p)
            Ki = K[rand_i,:]
            Kii = Ki[:,rand_i]
            x = istim[rand_i]
            y1 = np.sign(np.cos(2*x))
            y2 = np.sign(np.cos(6*x))
            #y1 = u[rand_i,0]
            #y2 = u[rand_i,5]
            yhat1 = Ki.T @ np.linalg.inv(Kii + 1/K.shape[0]*lamb*np.eye(p)) @ y1
            yhat2 = Ki.T @ np.linalg.inv(Kii + 1/K.shape[0]*lamb*np.eye(p)) @ y2

            yhat1_te = yhat1[~rand_i]
            yhat2_te = yhat2[~rand_i]

            err_lamb_easy[i,n] = np.mean( (y1_tot[~rand_i] - yhat1_te )**2 )
            err_lamb_hard[i,n] = np.mean( (y2_tot[~rand_i] - yhat2_te )**2 )

            rand_i = np.random.randint(0,K_avg.shape[0],p)
            Ki = K_avg[rand_i,:]
            Kii = Ki[:,rand_i]
            #x = istim[rand_i]
            x = stim_vals[rand_i]

            y1 = np.sign(np.cos(2*x))
            y2 = np.sign(np.cos(6*x))
            #y1 = u[rand_i,0]
            #y2 = u[rand_i,5]
            yhat1 = Ki.T @ np.linalg.inv(Kii + 1/K.shape[0]*lamb*np.eye(p)) @ y1
            yhat2 = Ki.T @ np.linalg.inv(Kii + 1/K.shape[0]*lamb*np.eye(p)) @ y2

            #yhat1_te = yhat1[~rand_i]
            #yhat2_te = yhat2[~rand_i]

            err_lamb_easy_avg[i,n] = np.mean( (y_easy_true - yhat1 )**2 )
            err_lamb_hard_avg[i,n] = np.mean( (y_hard_true - yhat2 )**2 )


    plt.figure(figsize=(1.8,1.5))
    plt.semilogx(lamb_vals, np.mean(err_lamb_easy, axis =1), color = 'blue')
    plt.semilogx(lamb_vals, np.mean(err_lamb_hard, axis =1), color = 'red')
    plt.semilogx(lamb_vals, np.mean(err_lamb_easy_avg, axis =1), '--', color = 'blue')
    plt.semilogx(lamb_vals, np.mean(err_lamb_hard_avg, axis =1), '--', color = 'red')
    plt.xlabel(r'$\lambda$',fontsize=myaxis_font)
    plt.ylabel(r'$E_g$', fontsize=myaxis_font)
    plt.tight_layout()
    plt.savefig(fig_dir + 'noisy_code_optimal_lamb.pdf')
    plt.show()

    """
    num_repeats=8
    #noise_vals = np.logspace(-3,-1,3)
    #noise_vals = [0.0, 0.001, 0.01]
    pvals = np.linspace(10, 3000, 20).astype('int')
    Nvals = [200,500,1000,2000]
    err_noisy = np.zeros((len(Nvals), len(pvals), num_repeats))
    lamb = 100.0
    for j,N in enumerate(Nvals):
        small_resp = sresp[0:N,:]
        #code_j = small_resp + sigma * np.random.standard_normal(small_resp.shape)
        code_j = small_resp
        Kj = 1/code_j.shape[0] * code_j.T @ code_j
        for n in range(num_repeats):
            print("n = %d" % n)
            for i,p in enumerate(pvals):

                rand_i = np.random.randint(0,Kj.shape[0],p)

                Ki = Kj[rand_i,:]
                Kii = Ki[:,rand_i]
                x = istim[rand_i]
                y1 = np.sign(np.cos(2*x))
                #y2 = np.sign(np.cos(6*x))
                #y1 = u[rand_i,0]
                #y2 = u[rand_i,5]
                yhat1 = Ki.T @ np.linalg.inv(Kii + 1/K.shape[0]*lamb*np.eye(p)) @ y1
                #yhat2 = Ki.T @ np.linalg.inv(Kii + 1/K.shape[0]*lamb*np.eye(p)) @ y2

                yhat1_te = yhat1[~rand_i]
                #yhat2_te = yhat2[~rand_i]

                err_noisy[j,i,n] = np.mean( (y1_tot[~rand_i] - yhat1_te )**2 )
                #err_hard[i,n] = np.mean( (y2_tot[~rand_i] - yhat2_te )**2 )

    plt.figure(figsize=(1.8,1.5))
    for j,N in enumerate(Nvals):
        plt.plot(pvals, np.mean(err_noisy[j,:,:], axis = 1), color = 'C%d' % j, label = r'$N = %d$' % N)

    #plt.plot(pvals, np.mean(err_hard, axis = 1), color = 'red',  label = 'High Freq.')
    plt.xlabel(r'$P$',fontsize=myaxis_font)
    plt.ylabel(r'$E_g$',fontsize=myaxis_font)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir + 'noisy_code_vary_N_learning_curves.pdf')
    plt.show()
    """

    p = 500

    num_repeats=10
    Nvals = [200,500,1000,2000]
    lamb_vals = np.logspace(-1,2.5,10)
    err_noisy_vs_lamb = np.zeros((len(Nvals), len(lamb_vals), num_repeats))
    for j, N in enumerate(Nvals):
        small_resp = sresp[0:N,:]
        code_j = small_resp
        Kj = 1/code_j.shape[0] * code_j.T @ code_j
        s = np.linalg.eigvalsh(Kj)
        Kj=Kj/np.amax(s)
        for k,lamb in enumerate(lamb_vals):
            for n in range(num_repeats):
                rand_i = np.random.randint(0,K.shape[0],p)
                Ki = Kj[rand_i,:]
                Kii = Ki[:,rand_i]
                x = istim[rand_i]
                y1 = np.sign(np.cos(2*x))
                yhat1 = Ki.T @ np.linalg.inv(Kii + 1/K.shape[0]*lamb*np.eye(p)) @ y1
                yhat1_te = yhat1[~rand_i]
                #yhat2_te = yhat2[~rand_i]

                err_noisy_vs_lamb[j,k,n] = np.mean( (y1_tot[~rand_i] - yhat1_te )**2 )

    plt.figure(figsize=(1.8,1.5))
    for j,N in enumerate(Nvals):
        plt.semilogx(lamb_vals, np.mean(err_noisy_vs_lamb[j,:,:], axis = 1), color = 'C%d' % j, label = r'$N = %d$' % N)
    #plt.plot(pvals, np.mean(err_hard, axis = 1), color = 'red',  label = 'High Freq.')
    plt.xlabel(r'$\lambda$',fontsize=myaxis_font)
    plt.ylabel(r'$E_g$',fontsize=myaxis_font)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir + 'noisy_code_vary_N_lamb_learning_curves.pdf')
    plt.show()
