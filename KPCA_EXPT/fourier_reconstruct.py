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
resp = (resp - mu) / sd
spont = (spont - mu) / sd
sv,u = eigsh(spont.T @ spont, k=32)
resp = resp - (resp @ u) @ u.T

# mean center each neuron
resp -= resp.mean(axis=0)
#resp = resp / (resp.std(axis = 0)+1e-6)


A_expt = imgs[:,90:180, istim]
#A_expt = transform.resize(A_expt, (A_expt.shape[0]//2, A_expt.shape[1]//2, A_expt.shape[2]))
A_expt = A_expt - np.mean(A_expt, axis = 2)[:,:,np.newaxis]
A_blurred = filt.gaussian(A_expt, sigma = 1.25)
A_blurred = A_blurred - A_blurred.mean(axis =2)[:,:,np.newaxis]
theta = np.linspace(0., 180., max(A_expt.shape))

def get_wavelet_decomp(image, level = 2):
    c = image
    wavelet = pywt.Wavelet('db1')
    level = pywt.dwt_max_level(min(c.shape[0], c.shape[1]), wavelet.dec_len)
    ans = pywt.wavedec2(c, wavelet, mode = 'sym', level = level)
    arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    plt.imshow(arr, cmap = 'gray_r')
    plt.show()
    arr = []
    for coeffs in ans:
        if not isinstance(coeffs, tuple):
            coeffs = (coeffs,)
        for w in coeffs:
            w_flat = w.flatten()
            #arr += [w_flat]
            arr += [float(np.mean(w_flat)), float(np.std(w_flat))]
    arr = np.array(arr)
    return arr

def get_wavelet_approx(images):
    a, _ = pywt.dwt2(images[:,:,0], wavelet = 'db1')
    result = np.zeros((a.shape[0], a.shape[1], images.shape[2]))
    for i in range(images.shape[2]):
        result[:,:,i] = pywt.dwt2(images[:,:,i], wavelet = 'db1')[0]
    return result

def get_gabor_basis(num_filters, xdim, ydim, sigma, k_std):
    xpos = np.linspace(1,xdim, xdim)
    ypos = np.linspace(1,ydim, ydim)
    gabors = np.zeros((xdim, ydim, num_filters))
    x_cen = xdim*np.random.random_sample(num_filters)
    y_cen = ydim*np.random.random_sample(num_filters)
    kvals = np.random.standard_normal((num_filters,2))
    for n in range(num_filters):
        phi = np.random.random_sample()
        Dx = (xpos - x_cen[n])**2
        Dy = (ypos - y_cen[n])**2

        K_x = kvals[n,0]*(xpos-x_cen[n])
        K_y = kvals[n,1]*(ypos-y_cen[n])
        D_mat = Dx[:,np.newaxis] + Dy[np.newaxis,:]
        K_mat  = K_x[:,np.newaxis] + K_y[np.newaxis,:]
        gabors[:,:,n] = np.exp(-0.5/sigma* D_mat )*np.cos(K_mat - phi)
    return gabors

def fourier_bandpass(images, k_frac, c = 0.2):
    #FT = np.fft.rfft2(image)
    all_rec = np.zeros(images.shape)
    for i in range(images.shape[2]):
        image= images[:,:,i]
        image = image - image.mean()
        FT = np.fft.fftn(image)

        kmax = k_frac * np.amin(FT.shape)
        kmin = np.sqrt(max(0,kmax**2 - c**2 * np.amin(FT.shape)**2))

        center = np.array( [ FT.shape[0]/2, FT.shape[1]/2] )
        grid1 = np.linspace(1,FT.shape[0], FT.shape[0])
        grid2 = np.linspace(1,FT.shape[1], FT.shape[1])

        X = np.zeros(( grid1.shape[0],grid2.shape[0],2 ))
        X[:,:,0] = grid1[:,np.newaxis]
        X[:,:,1] = grid2[np.newaxis,:]
        #Xc = X - center[np.newaxis,np.newaxis,:]
        D1 = np.linalg.norm( X , axis = 2)
        D2 = np.linalg.norm( X-X[-1,0,:][np.newaxis,np.newaxis,:], axis = 2 )
        D3 = np.linalg.norm( X-X[0,-1,:][np.newaxis,np.newaxis,:], axis = 2 )
        D4 = np.linalg.norm( X-X[-1,-1,:][np.newaxis,np.newaxis,:], axis = 2 )

        F2 =  FT * (D1 < kmax)*(D1>kmin) + FT * (D2 < kmax)*(D2>kmin) + FT * (D3 < kmax)*(D3>kmin) + FT * (D4 < kmax)*(D4>kmin)
        Xrec = np.fft.ifftn(F2)
        all_rec[:,:,i] = np.real(Xrec)
    return all_rec

cutoffs = [0.2, 0.25, 0.3]
all_vk_c = []
all_err_c = []
all_theory_c = []

cvals = [0.15,0.2,0.25]

plt.figure(figsize=(1.8,1.5))
plt.imshow(A_expt[:,:,0],cmap = 'gray')
plt.title(r'Original')
plt.axis('off')
plt.savefig('frog_img_original.pdf')
plt.show()

for c in cvals:
    all_vk = []
    all_err = []
    all_theory = []
    for t in cutoffs:
        A_fourier = fourier_bandpass(A_expt, t, c = c)
        A_fourier_flat = A_fourier.reshape((A_fourier.shape[0]*A_fourier.shape[1], A_fourier.shape[2]))
        y_fourier = A_fourier_flat.T
        y_fourier = y_fourier / np.sqrt( np.sum( y_fourier**2 ) )
        pvals = np.logspace(0,3.0, 8).astype('int')
        ptheory = np.logspace(0,3.5, 40)

        #plt.figure(figsize=(1.8,1.5))
        #plt.imshow(A_fourier[:,:,0],cmap = 'gray')
        #plt.title(r'$s_{max} = %0.2f$' % t)
        #plt.axis('off')
        #plt.savefig('frog_img_s_%0.2f.pdf' % t)
        #plt.show()

        err, theory, s_fourier, vk = get_all_task_info(resp, y_fourier, pvals, ptheory)
        all_err += [err]
        all_vk += [vk]
        all_theory += [theory]

    all_vk_c += [all_vk]
    all_err_c +=[all_err]
    all_theory_c += [all_theory]

for j,c in enumerate(cvals):
    plt.figure(figsize=(1.8,1.5))
    for i,t in enumerate(cutoffs):
        err_expt = all_err_c[j][i]
        plt.loglog(ptheory, all_theory_c[j][i].sum(axis = 0), color = 'C%d' % i)
        plt.errorbar(pvals, err_expt.mean(axis = 1), err_expt.std(axis=1)/err_expt.mean(axis=1), fmt='o', markersize = 2.5, color = 'C%d' % i)
    plt.xlabel(r'$P$',fontsize=myaxis_font)
    plt.ylabel(r'$E_g$',fontsize=myaxis_font)
    plt.tight_layout()
    plt.savefig('reconstruction_eg_fourier_c%0.2f.pdf' % c)
    #plt.show()


for j,c in enumerate(cvals):
    plt.figure(figsize=(1.8,1.5))

    for i, vk in enumerate(all_vk_c[j]):
        cumvk = np.cumsum(vk) / np.sum(vk)
        plt.plot(cumvk, label = r'$s_{max} = %0.2f$' % cutoffs[i])
    plt.legend()
    plt.xlabel(r'$k$',fontsize=myaxis_font)
    plt.ylabel(r'$C(k)$',fontsize=myaxis_font)
    plt.ylim([1e-4,1])
    plt.tight_layout()
    plt.savefig('cum_task_spec_nat_image_fourier_c%0.2f.pdf' % c)
    #plt.show()


"""
plt.figure(figsize=(1.8,1.5))
plt.loglog(np.linspace(1,len(s),len(s)),s)
plt.xlabel(r'$k$',fontsize=myaxis_font)
plt.ylabel(r'$\lambda_k$',fontsize=myaxis_font)
plt.tight_layout()
plt.savefig('spec_nat_image_recon.pdf')
plt.show()

axs =[plt.subplot(4,6,i+1) for i in range(24)]
for i, a in enumerate(axs):
    a.imshow(yhat[i,:].reshape((A_expt.shape[0], A_expt.shape[1])), cmap = 'gray')
    a.axis('off')
plt.tight_layout()
plt.subplots_adjust(wspace = 0.05,hspace=-0.5)
plt.savefig('reconstr_imgs.pdf')
plt.show()

axs =[plt.subplot(4,6,i+1) for i in range(24)]
for i, a in enumerate(axs):
    a.imshow(y[i,:].reshape((A_expt.shape[0], A_expt.shape[1])), cmap = 'gray')
    a.axis('off')
plt.tight_layout()
plt.subplots_adjust(wspace = 0.05,hspace=-0.5)
plt.savefig('ground_truth_imgs.pdf')
plt.show()



eig_fns = 1/np.sqrt(vec.shape[0]) * A_flat @ vec
eig_blur = 1/np.sqrt(vec.shape[0]) * A_blur_flat @ vec
print("total power")
print( np.sum(eig_fns**2) )

axs =[plt.subplot(4,6,i+1) for i in range(24)]
for i, a in enumerate(axs):
    a.imshow(eig_fns[:,i].reshape((A_expt.shape[0], A_expt.shape[1])), cmap = 'coolwarm')
    a.axis('off')
plt.tight_layout()
plt.subplots_adjust(wspace = 0.05,hspace=-0.5)
plt.savefig('eig_fns_nat_scenes.pdf')
plt.show()


axs =[plt.subplot(4,6,i+1) for i in range(24)]
for i, a in enumerate(axs):
    a.imshow(RevCorr_img[:,:,np.random.randint(resp.shape[1])], cmap = 'coolwarm')
    a.axis('off')
plt.tight_layout()
plt.subplots_adjust(wspace = 0.05,hspace=-0.5)
plt.savefig('rev_corr_nat_scenes.pdf')
plt.show()

vk = np.sum( eig_fns**2, axis = 0 )
vk_blur = np.sum(eig_blur**2, axis = 0)
plt.figure(figsize=(1.8,1.5))
plt.loglog(np.linspace(1,len(vk),len(vk)), vk)
plt.loglog(np.linspace(1,len(vk),len(vk)), vk_blur)
plt.xlabel(r'$k$',fontsize=myaxis_font)
plt.ylabel(r'$v_k^2$',fontsize=myaxis_font)
plt.ylim([1e-4,1])
plt.tight_layout()
plt.savefig('task_spec_nat_image_recon.pdf')
plt.show()
"""
