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
import scipy.optimize
import csv
import power_law
import matplotlib as mpl
from cycler import cycler
import scipy.linalg
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

# get the tuning curve r_i(theta-theta_i) = psi( cosh(s cos(theta))/cosh(s) - a )
#  r_i ~ psi( cosh(s * cos(theta))/cosh(s) - a )
def psi(x, cos_theta, f):
    b,q,s = x
    theta_star = math.pi/2 * f
    a = np.cosh(s * np.cos(theta_star)) / np.cosh(s)
    ri = np.cosh(s*cos_theta)/np.cosh(s) - a
    ri = ri* (ri>0)
    return b * ri**q

def get_K(x, *args):
    cos_mat = args[0]
    cos_vec = args[1]
    f = args[3]
    psi_mat = psi(x, cos_mat, f)
    psi_vec = psi(x, cos_vec, f)
    return 1/cos_vec.shape[0] * psi_mat @ psi_vec

# return params x data matrix
def grad_psi(x, cos_theta, f):
    b,q,s = x
    # lots of code here
    theta_star = math.pi/2*f
    ri = ( np.cosh(s*cos_theta) - np.cosh(s*np.cos(theta_star)) )/np.cosh(s)
    ans = np.zeros((len(x), len(cos_theta)))
    grad_s = (cos_theta*np.sinh(ri)-np.cos(theta_star)*np.sinh(ri)-np.tanh(s)*(np.cosh(s*cos_theta)-np.cosh(s*np.cos(theta_star))))/np.cosh(s)
    psi_vec = psi(x, cos_theta, f)
    ans[0,:] = 1/b * psi_vec
    for i in range(len(ri)):
        if ri[i] > 0:
            ans[1,i] = np.log(ri[i]) * ri[i]**q
            ans[2,i] = q*ri[i]**(q-1) * grad_s[i]
    return ans

def grad_K(x, *args):

    cos_mat = args[0]
    cos_vec = args[1]
    f = args[3]
    grad_psi_vec = grad_psi(x, cos_vec, f)
    psi_mat = psi(x, cos_mat, f)
    return 2*grad_psi_vec @ psi_mat

def loss_grad(x, *args):
    Ktrue = args[2]
    Khat = get_K(x, *args)

    diff = (Khat - Ktrue)
    # params x data
    g_K = grad_K(x,*args)
    gl = 1/len(diff) * g_K @ diff
    print("loss grad")
    print(gl)
    return g_K @ diff

def loss(x, *args):
    Ktrue = args[2]
    Khat = get_K(x, *args)
    myl = 0.5 * np.mean( (Ktrue-Khat)**2 )
    print(myl)
    return myl


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

#num_stim = 50
num_stim= 80
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
    sresp, istim, itrain, itest = utils.compile_resp(dat, npc=npc, normalize = False)

    percent_zeros = np.mean((sresp>0))
    print("Percent zeros model")
    print(percent_zeros)
    f_sp = 1-percent_zeros

    # need to keep f propto



    print(sresp.shape)

    # do some trial averaging to smooth out the responses...

    stim_vals = np.linspace(0, math.pi, num_stim)
    resp_avg = np.zeros( (sresp.shape[0], num_stim) )
    density = np.zeros( len(stim_vals))
    istim = istim % math.pi

    for i in range(num_stim-1):
        stim_inds = [j for j in range(len(istim)) if istim[j] >= stim_vals[i] and istim[j] < stim_vals[i+1]]
        resp_avg[:,i] = np.mean( sresp[:,stim_inds] , axis = 1)
        density[i] = len(stim_inds)

    print("percent zero resp avg")
    print(np.mean( (resp_avg>0) ) )
    #resp_avg = resp_avg[:,0:resp_avg.shape[1]-1]
    #stim_vals = stim_vals[0:stim_vals.shape[0]-1]


    cos_features = np.zeros((len(stim_vals), len(stim_vals)))
    for k in range(len(stim_vals)):
        cos_features[k,:] = np.cos(2*k*stim_vals)

    K = 1/resp_avg.shape[0] * resp_avg.T @ resp_avg
    kavg = np.zeros(K.shape[0])
    for n in range(K.shape[0]):
        for i in range(K.shape[0]):
            if i+n < K.shape[0]:
                kavg[n] += 1/K.shape[0]**2 * K[i,i+n]
            else:
                kavg[n] += 1/K.shape[0]**2 * K[i,i-n]
    kavg = kavg + kavg[::-1]
    kavg = kavg-np.amin(kavg)

    #plt.plot(stim_vals, kavg)
    #plt.xlabel(r'$\theta$',fontsize=20)
    #plt.ylabel(r'$K(\theta)$',fontsize=20)
    #plt.tight_layout()
    #plt.show()

    cos_vec = np.cos(stim_vals)
    cos_mat = cos_vec[:,np.newaxis] - cos_vec[np.newaxis,:]
    l = len(stim_vals)
    sub = 1
    #sub = 6
    args = (cos_mat[sub:l-sub,sub:l-sub], cos_vec[sub:l-sub], kavg[sub:l-sub], f_sp)
    x0 = [2, 2, 3]
    constrs = {sp.optimize.LinearConstraint( np.eye(3), lb = np.array([1e-2, 0.1, 1]), ub = np.array([25, 5, 5])  , keep_feasible = True) }
    #result = sp.optimize.minimize(loss, x0, method = 'Powell', args = args, bounds = Bounds, tol = 1e-12, options = {'maxiter': 2000, 'disp': True})
    result = sp.optimize.minimize(loss, x0, method = 'trust-constr', args = args, constraints = constrs, tol = 1e-12, options = {'maxiter': 10000, 'disp': True})
    #result = sp.optimize.minimize(loss, x0, method = 'Newton-CG', jac= loss_grad, args = args, tol = 1e-12, options = {'maxiter': 10000, 'disp': True})
    x = result.x
    success = result.success
    print(success)
    print("learned parameters")
    print(x)
    K_fit=get_K(x, *args)


    plt.figure(figsize=(1.8,1.5))
    plt.plot(stim_vals[sub:l-sub], K_fit, label = r'$q = %0.1f , s = %0.1f$' % (x[1],x[2]))
    plt.plot(stim_vals[sub:l-sub], kavg[sub:l-sub], 'o', markersize=1.5, color = 'black', label = 'Expt.')
    plt.xlabel(r'$\theta$',fontsize=myaxis_font)
    plt.ylabel(r'$K(\theta)$',fontsize=myaxis_font)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/fit_params.pdf')
    plt.show()

    # get spectra by computing the eigendecomposition of circulant matrix
    K1 = sp.linalg.circulant(kavg[sub:l-sub])
    K2 = sp.linalg.circulant(K_fit)
    print(K1.shape)
    s1 = np.sort(np.linalg.eigvalsh(K1))[::-1]
    s2 = np.sort(np.linalg.eigvalsh(K2))[::-1]

    plt.loglog(s1[0:len(s1)-3])
    plt.loglog(s2[0:len(s2)-3])
    plt.savefig('figures/spec_compare.pdf')
    plt.show()



    b,q,s = x
    a = np.cosh( s* np.cos(f_sp * math.pi/2 )) / np.cosh(s)
    print("q,s,a = %0.1f , %0.1f , %0.1f  " % (q,s,a))

    zvals = np.linspace(-1,1, 500)
    plt.figure(figsize=(1.8,1.5))
    diff=zvals - a
    plt.plot(zvals, np.maximum(0, zvals-a)**q, label = r'Fit')
    plt.plot(zvals, np.maximum(zvals-a, 0)**(0.5) , label = r'$q=0.5$')
    plt.plot(zvals, np.maximum(zvals - a, 0)**3, label = r'$q=3$')
    plt.plot(zvals, np.maximum(zvals,0)**q, label=r'$a = 0$')
    plt.title(r'Model Fit: $\hat{q},\hat{a}=%0.1f, %0.1f$' % (q,a), fontsize=myaxis_font)
    plt.xlabel(r'$z$', fontsize=myaxis_font)
    plt.ylabel(r'$g(z)$', fontsize=myaxis_font)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/nonlinearity_plots_fit.pdf')
    plt.show()

    # compute spectra for expt fit, and some other ones

    # q = 0, q = 1.3, q = 2  ,   s = 5 , a =
    # a =


    # a = cosh(s cos(f pi/2))/cosh(s) => f = 2/pi * cos^{-1}( 1/s )
    P = 250
    theta_vals = np.linspace(0, math.pi, P)
    theta_vals_mat = theta_vals[:,np.newaxis] - theta_vals[np.newaxis,:]

    stim_right_inds = [i for i,s in enumerate(stim_vals) if s<math.pi/2]
    stim_right = stim_vals[stim_right_inds]
    stim_left = - stim_right[::-1]

    args = (np.cos(theta_vals_mat), np.cos(theta_vals), 0 , f_sp)
    K = get_K(x, *args)
    K2 = get_K([1, 0.5, 5], *args)
    K3 = get_K([1,3,5], *args)

    args_4 = (np.cos(theta_vals_mat), np.cos(theta_vals), 0 , 1)
    K4 = get_K(x, *args_4)
    f5 = 2/math.pi * np.arccos( 1/s * np.arccosh(0.4 * np.cosh(s))  )
    args_5 = (np.cos(theta_vals_mat), np.cos(theta_vals), 0 , f5)
    K5 = get_K(x, *args_5)
    plt.figure(figsize=(1.8,1.5))


    stim_right = stim_vals[sub:int(l/2)]
    plt.plot(stim_right, K_fit[:len(stim_right)]/np.amax(K_fit))
    plt.plot(theta_vals[0:int(P/2)], K2[0:int(P/2)]/np.amax(K2))
    plt.plot(theta_vals[0:int(P/2)], K3[0:int(P/2)]/np.amax(K3))
    plt.plot(theta_vals[0:int(P/2)], K4[0:int(P/2)]/np.amax(K4))
    plt.plot(stim_right, kavg[sub:len(stim_right)+sub]/kavg[sub], 'o', markersize=1.5, color = 'black')
    plt.title('Kernels', fontsize=myaxis_font)
    plt.xlabel(r'$\theta$', fontsize=myaxis_font)
    plt.ylabel(r'$K(\theta)$', fontsize=myaxis_font)
    plt.tight_layout()
    plt.savefig('figures/kernel_plots_fit.pdf')
    plt.show()


    all_K = [K1, K2, K3, K4]
    plt.figure(figsize=(1.8,1.5))
    for i, Ki in enumerate(all_K):
        circ = sp.linalg.circulant(Ki)
        si = np.sort(np.linalg.eigvalsh(circ))[::-1]
        plt.loglog(si[0:100]/si[0])
    plt.loglog(s1/s1[0],'o', markersize=1.5, color = 'black')
    plt.title('Spectra', fontsize=myaxis_font)
    plt.xlabel(r'$k$', fontsize=myaxis_font)
    plt.ylabel(r'$\lambda_k$', fontsize=myaxis_font)
    plt.ylim([1e-8,10])
    plt.tight_layout()
    plt.savefig('figures/spectra_plots_fit.pdf')
    plt.show()
