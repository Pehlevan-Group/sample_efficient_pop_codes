import numpy as np
import scipy as sp
import scipy.io
from scipy.sparse.linalg import eigsh
import utils
import matplotlib.pyplot as plt
import power_law
import time
import matplotlib as mpl
from cycler import cycler
import skimage
from skimage import filters as filt
from skimage import transform
import pywt

mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

#import decoder
myfont = 6
myaxis_font = 8
plt.rcParams.update({'font.size': myfont})
line_width = 1


def sorted_spectral_decomp(resp, imgs=None):
    K = 1/resp.shape[1] * resp @ resp.T
    s,v = np.linalg.eigh(K)
    indsort = np.argsort(s)[::-1]
    s = s[indsort]
    v = v[:,indsort]
    return K,s,v


def compute_learning_curves(K, y, pvals, lamb):
    #coeffs = 1/K.shape[0] * (u.T @ y)**2
    #theory_lc = power_law.mode_errs(pvals, s, coeffs, lamb).sum(axis = 0)
    num_repeats = 5
    err_expt = np.zeros((len(pvals), num_repeats ))
    for n in range(num_repeats):
        for i,p in enumerate(pvals):
            inds = np.random.choice(K.shape[0], p, replace = True)
            #test_i = permed[p:permed.shape[0]]
            Ki = K[inds,:]
            Kii = Ki[:,inds]
            #K_test = Ki[:,test_i]
            print(Ki.shape)
            yi = y[inds, :]
            yhat = Ki.T @ np.linalg.solve(Kii + 1/K.shape[0]*lamb*np.eye(p),  yi)
            err_expt[i,n] = np.sum( (y - yhat )**2 )
    return err_expt, yhat



def svm(resp, y, T, eta, train_inds, test_inds, lamb):


    train = np.zeros(T)
    test = np.zeros(T)
    w = 1e-10*np.random.standard_normal(resp.shape[1])
    signed_resp = np.outer(y, np.ones(resp.shape[1])) * resp
    for t in range(T):
        yhat = np.sign( resp @ w )
        inner = signed_resp @ w
        step0 = np.heaviside(-1*inner, 0.0)
        step = np.heaviside(np.ones(len(inner)) - inner, 0.0 )
        w += eta * signed_resp[train_inds,:].T @ step[train_inds] - eta*lamb*w
        train[t] = np.mean(step0[train_inds])
        test[t] = np.mean(step0[test_inds])
    return train, test

def get_all_task_info(resp,y,pvals,ptheory):

    #K = 1/resp.shape[1] * resp @ resp.T
    K, s, vec = sorted_spectral_decomp(resp)
    lamb = np.trace(K)
    teacher = np.sum( ( vec.T @  y)**2, axis = 1)
    err_expt, yhat = compute_learning_curves(K, y, pvals, lamb)
    theory = power_law.mode_errs(ptheory, s, teacher, lamb)
    return err_expt, theory, s, teacher


np.random.seed(0)

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

# stim is preprocessed
istim = istim[:,0]
nimg = istim.max() # these are blank stims (exclude them)
resp = resp[istim<nimg, :]
istim = istim[istim<nimg]
print(np.amax(istim))
print("istim.shape")
print(istim.shape)

print("neural response shape")
print(resp.shape)

# subtract spont (32D)
mu = spont.mean(axis=0)
sd = spont.std(axis=0) + 1e-6
"""
resp = (resp - mu) / sd
spont = (spont - mu) / sd
sv,u = eigsh(spont.T @ spont, k=32)
resp = resp - (resp @ u) @ u.T
"""
# mean center each neuron
resp -= resp.mean(axis=0)
#resp = resp / (resp.std(axis = 0)+1e-6)


A_expt = imgs[:,90:180, istim]

A_flat = np.reshape(A_expt, (A_expt.shape[0]*A_expt.shape[1], A_expt.shape[2]))
print("flattened imgs shape")
print(A_flat.shape)

u,s,v = np.linalg.svd(A_flat)
indsort = np.argsort(s)[::-1]
u = u[:,indsort]
v = v[indsort,:]
s = s[indsort]

plt.loglog(s)
plt.show()

# A is D x P
# y is K x P where K is the number of preserved PCs


def sorted_spectral_decomp(resp, imgs=None):
    K = 1/resp.shape[1] * resp @ resp.T
    s,v = np.linalg.eigh(K)
    indsort = np.argsort(s)[::-1]
    s = s[indsort]
    v = v[:,indsort]
    K = K / s[0]
    s = s/s[0]
    return K,s,v

def compute_learning_curves(K, y, pvals, lamb):
    #coeffs = 1/K.shape[0] * (u.T @ y)**2
    #theory_lc = power_law.mode_errs(pvals, s, coeffs, lamb).sum(axis = 0)
    num_repeats = 5
    err_expt = np.zeros((len(pvals), num_repeats ))
    for n in range(num_repeats):
        for i,p in enumerate(pvals):
            inds = np.random.choice(K.shape[0], p, replace = True)
            #test_i = permed[p:permed.shape[0]]
            Ki = K[inds,:]
            Kii = Ki[:,inds]
            #K_test = Ki[:,test_i]
            print(Ki.shape)
            yi = y[inds, :]
            yhat = Ki.T @ np.linalg.solve(Kii + 1/K.shape[0]*lamb*np.eye(p),  yi)
            err_expt[i,n] = np.sum( (y - yhat )**2 )
    return err_expt, yhat

def get_all_task_info(resp,y,pvals,ptheory):

    #K = 1/resp.shape[1] * resp @ resp.T
    K, s, vec = sorted_spectral_decomp(resp)
    lamb = np.trace(K)
    teacher = np.sum( ( vec.T @  y)**2, axis = 1)
    err_expt, yhat = compute_learning_curves(K, y, pvals, lamb)
    theory = power_law.mode_errs(ptheory, s, teacher, lamb)
    return err_expt, theory, s, teacher

A_100 = (u[:,0:100] * s[0:100]) @ v[0:100,:]
A_200 = (u[:,0:200] * s[0:200]) @ v[0:200,:]
A_500 = (u[:,0:500] * s[0:500]) @ v[0:500,:]

A_100_flat = A_100.reshape(A_expt.shape)
A_200_flat = A_200.reshape(A_expt.shape)
A_500_flat = A_500.reshape(A_expt.shape)

plt.figure()
for n in range(3):
    plt.subplot(4,3,n+1)
    plt.imshow(A_expt[:,:,n])
for n in range(3):
    plt.subplot(4,3,3+n+1)
    plt.imshow(A_100_flat[:,:,n])
for n in range(3):
    plt.subplot(4,3,6+n+1)
    plt.imshow(A_200_flat[:,:,n])
for n in range(3):
    plt.subplot(4,3,9+n+1)
    plt.imshow(A_500_flat[:,:,n])
plt.show()


#y = A_200.T
N = 500
resp_N = resp[:,0:N]
y = s[0:100,np.newaxis] * v[0:100,:]
y = y.T
y = y/np.sqrt(np.sum(y**2))
pvals = np.logspace(0,3.0, 8).astype('int')
ptheory = np.logspace(0,3.0, 40)
err, theory, s_fourier, vk = get_all_task_info(resp_N, y, pvals, ptheory)

plt.figure(figsize=(1.8,1.5))
plt.loglog(pvals, err.mean(axis = 1))
plt.loglog(ptheory, theory.sum(axis = 0))
plt.show()
