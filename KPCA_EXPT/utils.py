import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA


def compile_resp(dat, nskip=4, npc=20, normalize = True):
    istim = dat['istim']
    # split stims into test and train
    itest = np.zeros((istim.size,), np.bool)
    itest[::nskip] = 1
    itrain = np.ones((istim.size,), np.bool)
    itrain[itest] = 0
    itrain = itrain.nonzero()[0]
    itest = np.nonzero(itest)[0]

    print("minimum spike rate")
    print( np.amin(dat['sresp'].copy()))
    # subtract off spont PCs
    if normalize == False:
        return dat['sresp'].copy(), istim, itrain, itest
    sresp = (dat['sresp'].copy() - dat['mean_spont'][:,np.newaxis]) / dat['std_spont'][:,np.newaxis]
    if npc > 0:
        sresp = sresp - dat['u_spont'][:,:npc] @ (dat['u_spont'][:,:npc].T @ sresp)
    sresp = sresp[:,:istim.size]
    # zscore sresp across stimuli (so each neuron has mean 0 / std 1 responses)
    ssub0 = sresp.mean(axis=1)
    sstd0 = sresp.std(axis=1) + 1e-6
    sresp = (sresp - ssub0[:,np.newaxis]) / sstd0[:,np.newaxis]
    return sresp, istim, itrain, itest


def compile_resp_spont_shift(dat, nskip=4, npc=20, normalize = True):
    istim = dat['istim']
    # split stims into test and train
    itest = np.zeros((istim.size,), np.bool)
    itest[::nskip] = 1
    itrain = np.ones((istim.size,), np.bool)
    itrain[itest] = 0
    itrain = itrain.nonzero()[0]
    itest = np.nonzero(itest)[0]

    print("minimum spike rate")
    print( np.amin(dat['sresp'].copy()))
    # subtract off spont PCs
    if normalize == False:
        return dat['sresp'].copy(), istim, itrain, itest
    sresp = (dat['sresp'].copy() - dat['mean_spont'][:,np.newaxis]) / dat['std_spont'][:,np.newaxis]
    if npc > 0:
        sresp = sresp - dat['u_spont'][:,:npc] @ (dat['u_spont'][:,:npc].T @ sresp)
    sresp = sresp[:,:istim.size]
    # zscore sresp across stimuli (so each neuron has mean 0 / std 1 responses)
    ssub0 = sresp.mean(axis=1)
    sstd0 = sresp.std(axis=1) + 1e-6
    #sresp = (sresp - ssub0[:,np.newaxis]) / sstd0[:,np.newaxis]
    sresp = sresp/sstd0[:,np.newaxis]
    spont_shift = ssub0/ sstd0
    
    return sresp, spont_shift, istim, itrain, itest




def process_spectrum(s):
    spectrum = np.power(s,2)
    spectrum = np.sort(spectrum)[::-1]
    spectrum= spectrum / np.sum(spectrum)
    spectrum = [s for s in spectrum if s>1e-12]
    return spectrum

def plot_spectrum(spectrum):
    a,b = best_linear_regression(spectrum)
    n_vals = np.linspace(1,len(spectrum), len(spectrum))
    line = np.power(n_vals, a) * np.exp(b)
    spectrum = np.power(spectrum, 2)
    plt.loglog(spectrum)
    plt.loglog(n_vals, line, label = r'$alpha = %lf$' % np.abs(a))
    plt.xlabel('$n$')
    plt.ylabel('$\lambda_n$')
    plt.legend()
    plt.savefig('spectrum.pdf')
    plt.show()
    return


def get_powerlaw(ss, trange):
    ''' fit exponent to variance curve'''
    logss = np.log(np.abs(ss))
    print(logss.shape)
    print(trange.shape)
    print(np.newaxis)
    y = logss[trange][:,np.newaxis]
    trange += 1
    nt = trange.size
    x = np.concatenate((-np.log(trange)[:,np.newaxis], np.ones((nt,1))), axis=1)
    w = 1.0 / trange.astype(np.float32)[:,np.newaxis]
    b = np.linalg.solve(x.T @ (x * w), (w * x).T @ y).flatten()

    allrange = np.arange(0, ss.size).astype(int) + 1
    x = np.concatenate((-np.log(allrange)[:,np.newaxis], np.ones((ss.size,1))), axis=1)
    ypred = np.exp((x * b).sum(axis=1))
    alpha = b[0]
    return alpha,ypred

def shuff_cvPCA(X, nshuff=5):
    ''' X is 2 x stimuli x neurons '''
    print(X.shape)
    nc = min(1024, X.shape[1])
    ss=np.zeros((nshuff,nc))
    for k in range(nshuff):
        iflip = np.random.rand(X.shape[1]) > 0.5
        X0 = X.copy()
        X0[0,iflip] = X[1,iflip]
        X0[1,iflip] = X[0,iflip]
        print(X0.shape)
        ss[k]=cvPCA(X0)
    return ss

def cvPCA(X):
    ''' X is 2 x stimuli x neurons '''
    pca = PCA(n_components=min(1024, X.shape[1])).fit(X[0].T)
    u = pca.components_.T
    sv = pca.singular_values_

    xproj = X[0].T @ (u / sv)
    cproj0 = X[0] @ xproj
    cproj1 = X[1] @ xproj
    ss = (cproj0 * cproj1).sum(axis=0)
    return ss


def CCA(X):

    return
