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


    Q,R = np.linalg.qr(np.random.standard_normal((1000,1000)))
    rotate = Q @ resp_avg[0:1000,:]
    sigma = 0.1
    filter = np.exp(- 0.5 * (stim_vals-math.pi)**2 /sigma**2 )
    all_filtered = np.zeros((3,len(filter)))
    plt.figure(figsize=(1.8,1.5))
    for i in range(3):
        all_filtered[i,:] = np.convolve(filter, resp_avg[i,:], 'same')
        #plt.plot(stim_vals,all_filtered[i,:], linewidth = line_width)
        plt.plot(stim_vals, resp_avg[i,:], linewidth = line_width)
    plt.xlabel(r'$\theta$', fontsize=myaxis_font)
    plt.ylabel(r'$r(\theta)$', fontsize=myaxis_font)
    #plt.xticks([])
    #plt.yticks([])
    plt.xticks([0,math.pi],[r'$0$',r'$\pi$'])
    plt.title('Original', fontsize=myaxis_font)
    plt.tight_layout()
    plt.savefig('tuning_curves_mouse_r.pdf')
    plt.show()


    #fig = plt.figure(figsize=(1.8,1.5))
    fig = plt.figure()
    plt.rcParams.update({'font.size': 20})

    ax = fig.add_subplot(111, projection = '3d')
    #ax.plot(resp_avg[0,:], resp_avg[1,:], resp_avg[2,:], label = 'Original Code')
    ax.plot(all_filtered[0,:], all_filtered[1,:], all_filtered[2,:], linewidth = 3)
    np.random.seed(1)
    Q, _  = np.linalg.qr(np.random.standard_normal((3,3)))
    all_rot = Q @ all_filtered
    #r_rot = Q @ resp_avg[0:3,:]
    #ax.plot(r_rot[0,:], r_rot[1,:], r_rot[2,:], label = 'Rotated Code')
    ax.plot(all_rot[0,:], all_rot[1,:], all_rot[2,:], linewidth = 3, color = 'C2')
    ax.scatter([],[], label ='Ori.', color = 'C0')
    ax.scatter([],[], label = 'Rot.', color = 'C2')
    #plt.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel([])
    #ax.legend(loc=(0.5,0.5,0.5), frameon=0)
    ax.legend(loc = 'best', bbox_to_anchor=(0.8, 0.8, 0.3, 0.3))
    plt.title('Neural Space')
    #ax.set_xlabel(r'$r_1$', fontsize=myaxis_font)
    #ax.set_ylabel(r'$r_2$', fontsize=myaxis_font)
    #ax.set_zlabel(r'$r_3$', fontsize = myaxis_font)
    ax.set_xlabel(r'$r_1$', fontsize=30)
    ax.set_ylabel(r'$r_2$',fontsize=30)
    ax.set_zlabel(r'$r_3$',fontsize=30)
    #plt.tight_layout()
    plt.savefig('tuning_curves_mouse_r_3d.pdf')
    plt.show()

    plt.rcParams.update({'font.size': myfont})


    plt.figure(figsize=(1.8,1.5))

    for i in range(6):
        filtered = np.convolve(filter, resp_avg[i,:], 'same')
        #plt.plot(stim_vals,filtered)
        #plt.plot(stim_vals, filtered, linewidth = line_width)
        plt.plot(stim_vals, resp_avg[i,:], linewidth = line_width)
    plt.xlabel(r'$\theta$', fontsize=myaxis_font)
    plt.ylabel(r'$r(\theta)$', fontsize=myaxis_font)
    #plt.xticks([])
    #plt.yticks([])
    #plt.title('Original Code', fontsize=myaxis_font)
    plt.tight_layout()
    plt.savefig('tuning_curves_mouse_r_many.pdf')
    plt.show()


    plt.figure(figsize=(1.8,1.5))

    for i in range(3):
        filtered = np.convolve(filter, rotate[i,:], 'same')
        #plt.plot(stim_vals, filtered, linewidth=line_width)
        plt.plot(stim_vals, rotate[i,:], linewidth = line_width)
    plt.xlabel(r'$\theta$', fontsize=myaxis_font)
    plt.ylabel(r'$\tilde{r}(\theta)$', fontsize=myaxis_font)
    plt.title('Rotated',fontsize=myaxis_font)
    plt.xticks([0,math.pi],[r'$0$',r'$\pi$'])

    #plt.xticks([])
    #plt.yticks([])
    plt.tight_layout()
    plt.savefig('tuning_curves_mouse_rotated_r.pdf')
    plt.show()

    #K_sub = 1/1000 * resp_avg[0:1000,:].T @ resp_avg[0:1000,:]
    #K_sub_rotate = 1/1000 * rotate.T @ rotate
    K_sub = 1/resp_avg.shape[0] * resp_avg.T @ resp_avg
    plt.figure(figsize=(1.8,1.5))
    vmax = np.abs(K_sub).max()
    vmin = - vmax
    plt.imshow(K_sub, cmap = 'seismic', vmin = vmin, vmax = vmax)
    plt.xticks([0,resp_avg.shape[1]],[r'$0$',r'$\pi$'])
    plt.yticks([0,resp_avg.shape[1]],[r'$\pi$',r'$0$'])
    plt.xlabel(r'$\theta_1$', fontsize=myaxis_font)
    plt.ylabel(r'$\theta_2$', fontsize=myaxis_font)
    plt.title(r'Kernel',fontsize=myaxis_font)
    plt.colorbar()

    plt.tight_layout()
    plt.savefig('kernel_matrix_sub_no_rotate.pdf')
    plt.show()





    #Nvals = [10,20, 50, 100, 200, 500, 1000, 2000,5000, resp_avg.shape[0]]
    Nvals = np.logspace(1.5,np.log10(0.8*resp_avg.shape[0]-1), 50).astype('int')
    power = np.zeros(len(Nvals))
    num_subsample = 250
    num_eig = 10
    eigs = np.zeros((num_eig, len(Nvals)))

    """
    for i, Ni in enumerate(Nvals):
        print("N = %d" % Ni)
        for j in range(num_subsample):
            neuron_idsi = np.random.choice(resp_avg.shape[0], Ni, replace = False)
            r = resp_avg[neuron_idsi,:]
            power[i] += 1/num_subsample * np.mean( r**2 )
            u,s,v = np.linalg.svd( 1/np.sqrt(Ni) * r, full_matrices = False)
            s = np.sort(s)[::-1]
            eigs[:,i] += 1/num_subsample * s[0:10]**2

    plt.semilogx(Nvals, power)
    plt.ylim([0,2*np.amax(power)])
    plt.xlabel(r'$N$', fontsize = 20)
    plt.ylabel(r'$\frac{1}{N} \sum_{i=1}^N \left< r_i(x)^2 \right>_{x}$', fontsize = 20)
    plt.tight_layout()
    plt.savefig('power_vs_N.pdf')
    plt.show()

    for i in range(num_eig):
        plt.loglog(Nvals, eigs[i,:])
    plt.xlabel(r'$N$', fontsize = 20)
    plt.ylabel(r'$\lambda_k$', fontsize = 20)
    plt.tight_layout()
    plt.savefig('kernel_eigenvalues_vs_N.pdf')
    plt.show()
    """


    #resp_avg = sp.signal.fftconvolve( filter.reshape((1, filter.shape[0])), resp_avg, 'same')
    # compute kernel
    K = 1/resp_avg.shape[0] * resp_avg.T @ resp_avg
    plt.figure(figsize=(1.8,1.5))

    plt.imshow(K)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(r'$\theta_1$', fontsize=myaxis_font)
    plt.ylabel(r'$\theta_2$', fontsize=myaxis_font)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('kernel_matrix.pdf')
    plt.show()

    kavg = np.zeros(K.shape[0])
    for k in range(K.shape[0]):
        inds = np.roll(np.arange(K.shape[0]),-k)
        kavg += 1/K.shape[0] * K[k,inds]
    plt.figure(figsize=(1.8,1.5))

    plt.plot(stim_vals, kavg)
    plt.xlabel(r'$\theta$', fontsize=myaxis_font)
    plt.ylabel(r'$K(\theta)$', fontsize=myaxis_font)
    plt.tight_layout()
    plt.savefig('kernel_real_space.pdf')
    plt.show()
    kavg = np.zeros(K.shape[0])

    for k in range(K.shape[0]):
        inds = np.roll(np.arange(K.shape[0]),-k)
        kavg += 1/K.shape[0] * K[k,inds]

    stim2 = stim_vals[0:len(stim_vals)-1]
    kavg2 = kavg[0:len(stim_vals)-1]
    x_shift = np.heaviside(-1e-1+math.pi*np.ones(len(stim2)) - stim2, np.zeros(len(stim2)))*stim2
    x_shift += np.heaviside(1e-1-math.pi*np.ones(len(stim2)) + stim2, np.zeros(len(stim2))) * (stim2 - 2*math.pi*np.ones(len(stim2)))
    k2 = kavg2 + kavg2[::-1]
    print(k2)
    plt.figure(figsize=(1.8,1.5))
    plt.plot(x_shift, k2)
    plt.xlabel(r'$\theta$', fontsize=myaxis_font)
    plt.ylabel(r'$K(\theta)$', fontsize=myaxis_font)
    plt.tight_layout()
    plt.savefig('kernel_real_space_shift.pdf')
    plt.show()

    s,u = np.linalg.eigh(K)
    sort_inds = np.argsort(s)[::-1]
    u = u[:,sort_inds]
    s = s[sort_inds]
    plt.figure(figsize=(1.8,1.5))
    plt.loglog(s/s[0], linewidth = line_width)
    plt.xlabel(r'$k$', fontsize=myaxis_font)
    plt.ylabel(r'$\lambda_k$', fontsize=myaxis_font)
    plt.ylim([1e-3,1])
    plt.tight_layout()
    plt.savefig('spectrum_population_grating_big.pdf')
    plt.show()

    # smooth out eigenfunctions for visualization
    plt.figure(figsize=(2.25,1.5))

    sigma = 0.075
    filter = np.exp(- 0.5 * (stim_vals-math.pi)**2 /sigma**2 ) / np.sqrt(2*math.pi*sigma**2)

    for i in range(5):
        filtered = np.convolve(u[:,i],filter, 'same')
        plt.plot(stim_vals, u[:,i] + 0.5*i*np.ones(len(stim_vals)), label = 'k = %d' % i, linewidth=line_width)
    plt.legend(bbox_to_anchor = (1,1) )
    plt.xlabel(r'$\theta$', fontsize = myaxis_font)
    plt.ylabel(r'$\phi_k(\theta)$', fontsize=myaxis_font)
    plt.xticks([0,math.pi], [r'$0$',r'$\pi$'])
    #plt.xticks([])
    #plt.yticks([])
    plt.tight_layout()
    plt.savefig('eigenfunctions_big.pdf')
    plt.show()




    spectrum = s / s[0]
    pvals = np.logspace(0,4,300)
    me = power_law.mode_errs(pvals, spectrum, np.ones(len(s)), 10)
    inds = [1,10,20,50]
    plt.figure(figsize=(1.8,1.5))
    for i,ind in enumerate(inds):
        plt.loglog(pvals, me[ind-1,:] / me[ind-1,0], label = r'$k = %d$' % ind, linewidth = line_width)
    plt.legend()
    plt.xlabel(r'$p$', fontsize = myaxis_font)
    plt.ylabel(r'$E_k$', fontsize=myaxis_font)
    plt.title('Mode Errors',fontsize=myaxis_font)
    plt.tight_layout()
    plt.savefig('mode_err_curves_population_grating.pdf')
    plt.show()






    # u is the eigenvectors, K is
    feature_space = u.T @ K
    inds_0_90 = [i for i in range(len(stim_vals)) if np.cos(2*stim_vals[i]) > 0]
    inds_90_180 = [i for i in range(len(stim_vals)) if np.cos(2*stim_vals[i]) <= 0]
    plt.figure(figsize=(1.8,1.5))

    plt.scatter(feature_space[0,inds_0_90], feature_space[1, inds_0_90], s=1, color = 'C4', label = r'$+1$')
    plt.scatter(feature_space[0,inds_90_180], feature_space[1, inds_90_180], s=1, color = 'C5', label = r'$-1$')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('k-PC 1', fontsize=myaxis_font)
    plt.ylabel('k-PC 2', fontsize = myaxis_font)
    plt.title('Easy Task',fontsize=myaxis_font)
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_k_space_mouse_low_freq.pdf')
    plt.show()

    inds_0 = [i for i in range(len(stim_vals)) if np.cos(6*stim_vals[i]) > 0]
    inds_1= [i for i in range(len(stim_vals)) if np.cos(6*stim_vals[i]) <= 0]
    plt.figure(figsize=(1.8,1.5))

    plt.scatter(feature_space[0,inds_0], feature_space[1, inds_0], s = 1, color = 'C4',label = r'$+1$')
    plt.scatter(feature_space[0,inds_1], feature_space[1, inds_1], s=1, color = 'C5', label = r'$-1$')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('k-PC 1', fontsize=myaxis_font)
    plt.ylabel('k-PC 2', fontsize=myaxis_font)
    plt.title('Hard Task',fontsize=myaxis_font)
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_k_space_mouse_high_freq.pdf')
    plt.show()


    y1 = np.sign(np.cos(2*stim_vals))
    y2 = np.sign(np.cos(6*stim_vals))
    plt.figure(figsize=(1.8,1.5))

    plt.plot(stim_vals, y1, linewidth=line_width, color = 'C0')
    plt.plot(stim_vals, y2, '--', linewidth=line_width, color = 'C2')
    #plt.legend()
    plt.xlabel(r'$\theta$', fontsize=myaxis_font)
    plt.ylabel(r'$y(\theta)$', fontsize=myaxis_font)
    plt.title('Orientation Tasks',fontsize=myaxis_font)
    plt.xticks([0,math.pi], [r'$0$',r'$\pi$'])
    plt.tight_layout()
    plt.savefig('low_high_target_visual.pdf')
    plt.show()

    coeffs1 = (u.T @ y1)**2
    coeffs2 = (u.T @ y2)**2
    print(coeffs1[0:10])
    print(coeffs2[0:10])
    sort1 = np.argsort(coeffs1)[::-1]
    sort2 = np.argsort(coeffs2)[::-1]
    plt.figure(figsize=(1.8,1.5))

    plt.scatter(feature_space[sort1[0],inds_0_90], feature_space[sort1[1], inds_0_90], color = 'C4', s= 1)
    plt.scatter(feature_space[sort1[0],inds_90_180], feature_space[sort1[1], inds_90_180], color = 'C5', s= 1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('k-PC %d'% sort1[0], fontsize=myaxis_font)
    plt.ylabel('k-PC %d' % sort1[1], fontsize =myaxis_font)
    plt.tight_layout()
    plt.savefig('feature_k_space_mouse_low_freq_max_var_kpc.pdf')
    plt.show()

    inds_0 = [i for i in range(len(stim_vals)) if np.cos(5*stim_vals[i]) > 0]
    inds_1= [i for i in range(len(stim_vals)) if np.cos(5*stim_vals[i]) <= 0]
    plt.figure(figsize=(1.8,1.5))

    plt.scatter(feature_space[sort2[0],inds_0], feature_space[sort2[1], inds_0], color = 'C4', s = 1)
    plt.scatter(feature_space[sort2[0],inds_1], feature_space[sort2[1], inds_1], color = 'C5', s = 1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('k-PC %d'% sort1[0])
    plt.ylabel('k-PC %d' % sort1[1])
    plt.savefig('feature_k_space_mouse_high_freq_max_var_kpc.pdf')
    plt.show()



    plt.figure(figsize=(1.8,1.5))
    plt.scatter(feature_space[2,inds_0], feature_space[3, inds_0], color = 'C4', label = r'$+1$', s =1)
    plt.scatter(feature_space[2,inds_1], feature_space[3, inds_1], color = 'C5', label = r'$-1$', s = 1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('k-PC %d'% 3, fontsize=myaxis_font)
    plt.ylabel('k-PC %d' % 4, fontsize=myaxis_font)
    plt.title('Hard Task',fontsize=myaxis_font)
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_k_space_mouse_high_freq_68_kpc.pdf')
    plt.show()



    #R = np.random.standard_normal((resp_avg.shape[0], resp_avg.shape[0]))
    """
    Nr = 500
    R = sp.stats.ortho_group.rvs(Nr)
    #s, v = np.linalg.eig(R)
    rotated_code = R @ resp_avg[0:Nr,:]
    for i in range(3):
        plt.plot(stim_vals, resp_avg[i,:], color = 'C%d' % i)
        plt.plot(stim_vals, rotated_code[i,:], '--', color = 'C%d' % i)
    plt.show()
    """

    # kernel regression expt
    pvals = np.logspace(0.4,2, 12).astype('int')
    num_repeats = 30
    lamb = 12
    y_easy_true = np.sign(np.cos(2*stim_vals))
    y_hard_true = np.sign(np.cos(6*stim_vals))
    err_easy = np.zeros((len(pvals), num_repeats))
    err_hard = np.zeros((len(pvals), num_repeats))
    for n in range(num_repeats):
        for i,p in enumerate(pvals):
            rand_i = np.random.randint(0,K.shape[0],p)
            Ki = K[rand_i,:]
            Kii = Ki[:,rand_i]
            x = stim_vals[rand_i]
            y1 = np.sign(np.cos(2*x))
            y2 = np.sign(np.cos(6*x))
            #y1 = u[rand_i,0]
            #y2 = u[rand_i,5]
            yhat1 = Ki.T @ np.linalg.inv(Kii + 1/K.shape[0]*lamb*np.eye(p)) @ y1
            yhat2 = Ki.T @ np.linalg.inv(Kii + 1/K.shape[0]*lamb*np.eye(p)) @ y2
            err_easy[i,n] = np.mean( (y_easy_true - yhat1 )**2 )
            err_hard[i,n] = np.mean( (y_hard_true - yhat2 )**2 )

    plt.figure(figsize=(1.8,1.5))
    plt.plot(stim_vals, yhat1, label = r'Ori.', linewidth=line_width)
    plt.plot(stim_vals, yhat1, '--', color = 'C2', label = r'Rot.', linewidth=line_width)
    plt.plot(stim_vals, y_easy_true, '--', color = 'black', label = r'$y(x)$', linewidth=line_width)
    plt.xlabel(r'$\theta$', fontsize=myaxis_font)
    plt.ylabel(r'$y(\theta)$', fontsize=myaxis_font)
    plt.title('Target Function',fontsize=myaxis_font)
    plt.xticks([0,math.pi], [r'$0$',r'$\pi$'])

    #plt.xticks([])
    #plt.yticks([])
    #plt.legend()
    plt.tight_layout()
    plt.savefig('task_visual.pdf')
    plt.show()

    kmax = 50
    print(len(spectrum))
    print(np.sum(s))

    coeffs_easy = (1/K.shape[0] * u.T @ y_easy_true)**2
    coeffs_hard = (1/K.shape[0] * u.T @ y_hard_true)**2
    plt.figure(figsize=(1.8,1.5))
    plt.plot(np.cumsum(coeffs_easy)/np.sum(coeffs_easy), label = 'low freq.', linewidth=line_width, color = 'C0')
    plt.plot(np.cumsum(coeffs_hard)/np.sum(coeffs_hard), label = 'high freq.', linewidth=line_width, color = 'C2')
    plt.xlabel(r'$k$', fontsize=myaxis_font)
    plt.ylabel(r'$C(k)$', fontsize=myaxis_font)
    plt.title('Cumulative Power',fontsize=myaxis_font)
    plt.legend()
    plt.tight_layout()
    plt.savefig('cumulative_sign_harmonic_mouse.pdf')
    plt.show()

    plt.figure(figsize=(2.4,2))
    decay_coeff = 1-np.cumsum(coeffs_easy)/np.sum(coeffs_easy)
    decay_coeff = decay_coeff[0:95]
    len_i = len(decay_coeff)
    log_decay_coeff = np.log(decay_coeff)
    kvals_linsp= np.log( np.linspace(1,len_i, len_i) )
    a = (np.mean(kvals_linsp*log_decay_coeff) - np.mean(kvals_linsp)*np.mean(log_decay_coeff)) / (np.mean(kvals_linsp**2) - np.mean(kvals_linsp)**2)
    b = np.mean(decay_coeff) - a*np.mean(kvals_linsp)
    print("a val from fit %0.2f" % a)
    plt.loglog(decay_coeff, label ='Orientation Task')
    plt.loglog(np.linspace(1,len_i,len_i)**a * decay_coeff[0], '--', color = 'black', label = r'$k^{%.1f}$' % a)
    plt.xlabel(r'$k$', fontsize=myaxis_font)
    plt.ylabel(r'$1 - C(k)$', fontsize=myaxis_font)
    plt.title('Cumulative Power',fontsize=myaxis_font)
    plt.legend()
    plt.tight_layout()
    plt.savefig('cumulative_powerlaw_scaling.pdf')
    plt.show()



    bvals = [2,3,4]
    all_s = [s] + [np.linspace(1,len(s),len(s))**(-b) for b in bvals]
    plt.figure(figsize=(2.4,2))
    for i,si in enumerate(all_s):
        if i == 0:
            label = 'Expt.'
        else:
            label = r'$b = %d$' % bvals[i-1]
        plt.loglog(pvals, power_law.mode_errs(pvals,si,coeffs_easy,10).sum(axis = 0), label = label)
    plt.xlabel(r'$p$', fontsize=myaxis_font)
    plt.ylabel(r'$E_g$', fontsize=myaxis_font)
    plt.title('Learning Curves', fontsize=myaxis_font)
    #ax.set_xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig('power_law_scalings_mouse_ori.pdf')
    plt.show()

    plt.figure(figsize=(2.4,2))
    for i,si in enumerate(all_s):
        if i == 0:
            label = 'Expt.'
        else:
            label = r'$b = %d$' % bvals[i-1]
        plt.loglog(si/si[0], label = label)
    plt.xlabel(r'$k$', fontsize=myaxis_font)
    plt.ylabel(r'$\lambda_k$', fontsize=myaxis_font)
    plt.title('Power Law Spectra', fontsize=myaxis_font)
    plt.ylim([1e-5,2])
    #ax.set_xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig('power_law_spectra_mouse.pdf')
    plt.show()


    plt.figure(figsize=(2.4,2))
    plt.plot(stim_vals, y_easy_true, '--', color = 'black', label = r'$y(x)$', linewidth=line_width)
    plt.xlabel(r'$\theta$', fontsize=myaxis_font)
    plt.ylabel(r'$y(\theta)$', fontsize=myaxis_font)
    plt.title('Target Function',fontsize=myaxis_font)
    plt.xticks([0,math.pi], [r'$0$',r'$\pi$'])
    #plt.xticks([])
    #plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig('task_visual_powerlaw.pdf')
    plt.show()



    ptheory = pvals
    theory_easy = np.sum( power_law.mode_errs(ptheory, s, coeffs_easy, lamb), axis = 0)
    theory_hard = np.sum( power_law.mode_errs(ptheory, s, coeffs_hard, lamb), axis = 0)


    fig = plt.figure()
    ax = plt.axes()
    easy_mean = np.mean(err_easy, axis = 1)
    hard_mean = np.mean(err_hard, axis = 1)

    plt.figure(figsize=(1.8,1.5))

    plt.errorbar(pvals, easy_mean/easy_mean[0], np.std(err_easy, axis=1),   fmt = 'o', markersize=2.5, color = 'C0', linewidth=line_width)
    plt.errorbar(pvals, hard_mean/hard_mean[0], np.std(err_hard, axis = 1), fmt = 'o', markersize=2.5, color = 'C2', linewidth=line_width)
    #plt.plot(pvals, me[0,:] / me[0,0], '--', color = 'C0')
    #plt.plot(pvals, me[5,:] / me[5,0], '--', color = 'C1')
    plt.plot(ptheory, theory_easy/theory_easy[0], '--', color = 'C0', linewidth=line_width)
    plt.plot(ptheory, theory_hard/theory_hard[0], '--', color = 'C2', linewidth=line_width)
    plt.xlabel(r'$p$', fontsize=myaxis_font)
    plt.ylabel(r'$E_g$', fontsize=myaxis_font)
    plt.title('Learning Curves', fontsize=myaxis_font)
    #ax.set_xscale('log')
    #plt.legend()
    plt.tight_layout()
    plt.savefig('mouse_lc.pdf')
    plt.show()


    plt.figure(figsize=(1.8,1.5))
    plt.errorbar(pvals, easy_mean/easy_mean[0], np.std(err_easy,axis=1), fmt = 'o', markersize=2, color = 'C0', label = 'Original', linewidth=line_width)
    plt.errorbar(pvals, easy_mean/easy_mean[0], np.std(err_hard,axis=1), fmt = '^', markersize=2, color = 'C2', label = 'Rotated', linewidth=line_width)
    plt.plot(ptheory, theory_easy/theory_easy[0], '--', color = 'black', label = 'theory', linewidth=line_width)
    plt.legend()
    plt.xlabel(r'$p$', fontsize=myaxis_font)
    plt.ylabel(r'$E_g$', fontsize=myaxis_font)
    plt.title('Learning Curves',fontsize=myaxis_font)
    plt.tight_layout()
    plt.savefig('mouse_rotate_lcs.pdf')
    plt.show()
