import numpy as np
import scipy as sp
import scipy.optimize
import matplotlib.pyplot as plt

# can we verify the asymptotic scalings derived for power
# law decay kernels?

# let's solve this

def f(z, *args):
    p, spectrum, lamb = args
    return z - lamb - z*np.sum( spectrum/(p*spectrum+z) )

def fp(z, *args):
    p, spectrum, lamb = args
    return 1 - np.sum( spectrum/(p*spectrum+z) ) + np.sum(spectrum/(p*spectrum + z)**2)


def solve_implicit(pvals, spectrum, lamb):

    zvals = np.zeros(len(pvals))
    for i,p in enumerate(pvals):
        args = (p, spectrum, lamb)
        zvals[i] = sp.optimize.root_scalar(f = f, method = 'newton', fprime = fp, x0=2*np.sum(spectrum)+2*lamb,args = args).root
    return zvals


def gamma_fn(pvals, zvals, spectrum):
    all_vals = np.zeros(len(pvals))
    for i, p in enumerate(pvals):
        all_vals[i] = p*np.sum(spectrum**2/(spectrum * p + zvals[i])**2 )
    return all_vals

def learning_curves(pvals, spectrum, teacher, lamb):

    zvals = solve_implicit(pvals, spectrum, lamb)
    gamma = gamma_fn(pvals, zvals, spectrum)
    err = np.zeros(len(pvals))
    for j in range(len(spectrum)):
        lamb_j = spectrum[j]
        teacher_weights_j = teacher[j]
        err += zvals**2/(1-gamma) * teacher_weights_j/(lamb_j*pvals+zvals)**2
        #if j == len(spectrum)-1:
        #    plt.loglog(pvals[0:10], err[0:10])
        #    plt.show()
    return err


def mode_errs(pvals, spectrum, teacher, lamb):

    zvals = solve_implicit(pvals, spectrum, lamb)
    gamma = gamma_fn(pvals, zvals, spectrum)
    err = np.zeros( (len(spectrum), len(pvals)) )
    for j in range(len(spectrum)):
        lamb_j = spectrum[j]
        teacher_weights_j = teacher[j]
        err[j,:] = zvals**2/(1-gamma) * teacher_weights_j/(lamb_j*pvals+zvals)**2
        #if j == len(spectrum)-1:
            #plt.loglog(pvals[0:10], err[0:10])
            #plt.show()
    return err
