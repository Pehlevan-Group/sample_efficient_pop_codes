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
mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

fig_dir = 'figures/'


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


def compute_learning_curves(K, y, pvals, lamb):
    s,u = np.linalg.eigh(K)
    sort_inds = np.argsort(s)[::-1]
    u = u[:,sort_inds]
    s = s[sort_inds]
    coeffs = 1/K.shape[0] * (u.T @ y)**2
    theory_lc = power_law.mode_errs(pvals, s, coeffs, lamb).sum(axis = 0)
    num_repeats = 2
    err_expt = np.zeros((len(pvals), num_repeats ))
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
    return coeffs, s, theory_lc, err_expt



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
sv,u = eigsh(spont.T @ spont, k=32)
resp = resp - (resp @ u) @ u.T

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

plt.imshow(A[:,:,3], cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(fig_dir+'bird_picture.pdf')
plt.show()

plt.imshow(B[:,:,8].T, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(fig_dir+'mouse_picture.pdf')
plt.show()

y = class_stim[inds_12]
a = np.amin(y)
b = np.amax(y)

y = 2/(b-a)*(y-np.mean([a,b]))
print(y.shape)
print(y)

fontsize = 6
myaxis_font=8
line_width= 1
plt.rcParams.update({'font.size': fontsize})
resp_12 = resp[inds_12,:]
print("shape of responses after restricting the inds")
print(resp_12.shape)
K, s, v = sorted_spectral_decomp(resp_12, imgs)

plt.figure(figsize=(1.8,1.5))
plt.loglog(s, linewidth=line_width)
plt.xlabel(r'$k$', fontsize=myaxis_font)
plt.ylabel(r'$\lambda_k$', fontsize=myaxis_font)
plt.title('Spectra Natural Images', fontsize=myaxis_font)
plt.tight_layout()
plt.savefig(fig_dir+'spectrum_natural_image_task.pdf')
plt.show()


coeffs = (v.T @ y)**2


#plt.loglog(s)
#plt.show()
#pvals = np.logspace(1,np.log10(K.shape[0]-1),15).astype('int')
#lamb = 1e-8
#coeffs, s, theory, expt = compute_learning_curves(K, Y, pvals, lamb)
N = K.shape[0]


ck = np.cumsum(coeffs)/np.sum(coeffs)
plt.figure(figsize=(1.8,1.5))
plt.plot(ck, linewidth=line_width)
plt.xlabel(r'$k$', fontsize=myaxis_font)
plt.ylabel(r'$C(k)$', fontsize=myaxis_font)
plt.title('Image Recognition', fontsize=myaxis_font)
plt.tight_layout()
plt.savefig(fig_dir + 'cumulative_power_natural_image_task.pdf')
plt.show()

Psi = v @ np.diag(np.sqrt(s))
print(Psi.shape)

print("max inds")
print( max(inds1) )

pos = [i for i in range(len(y)) if y[i] == 1]
neg = [i for i in range(len(y)) if y[i] == -1]
plt.figure(figsize=(1.8,1.5))
plt.scatter(Psi[pos,0], Psi[pos,1], s = 0.3, color = 'C4', label = 'bird')
plt.scatter(Psi[neg,0], Psi[neg,1], s = 0.3, color = 'C5', label = 'mouse')
plt.xlabel(r'$\sqrt{\lambda_1} \psi_1(\theta)$', fontsize=myaxis_font)
plt.ylabel(r'$\sqrt{\lambda_2} \psi_2(\theta)$', fontsize = myaxis_font)
plt.title('Image Recognition',fontsize=myaxis_font)
plt.xticks([])
plt.yticks([])
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir + 'feature_space_natural_images.pdf')
plt.show()

pvals = np.linspace(10, 500, 20).astype('int')
s = s/s[0]
K = K/s[0]
lamb = 0.01

coeffs, s, theory_lc, err_expt = compute_learning_curves(K, y, pvals, lamb)

plt.plot(pvals, np.mean(expt, axis = 1))
plt.xlabel(r'$p$', fontsize = myaxis_font)
plt.ylabel(r'$E_g$', fontsize=myaxis_font)
plt.tight_layout()
plt.savefig(fig_dir + 'natural_image_classification.pdf')
plt.show()
