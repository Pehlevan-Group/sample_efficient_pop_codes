import numpy as np
import matplotlib.pyplot as plt
import theory_curve
import scipy as sp
import math
import scipy.special
import matplotlib as mpl
from cycler import cycler
from tqdm import tqdm

myaxis_font = 8
myfont = 6
mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

plt.rcParams.update({'font.size': myfont})
line_width = 1


# ok... so we want an expt where the bulk of the spectrum matters
#    this should be about the bandwidth of the kernel


def von_mises(x,xp, beta):
    D = np.outer(x, np.ones(xp.shape)) - np.outer(np.ones(x.shape[0]), xp)
    return np.exp(beta * (np.cos(D)-1))

def power_law_kernel(x,xp, b, km):
    D = np.outer(x, np.ones(xp.shape)) - np.outer(np.ones(x.shape[0]), xp)
    K = np.zeros(D.shape)
    for k in range(km):
        K += (1+5*k**2)**(-b) * np.cos(k*D)
    return K

def spectrum(beta, K):
    return sp.special.iv(np.linspace(1,K,K), beta)

def target_fn(x):
    return np.cos(x) - np.sin(2*x)

def target_k(x,k):
    return np.cos(k*x)

def k_expt(pvals, K, y, lamb, repeat = 30):

    X = np.linspace(-math.pi,math.pi, P)
    err = np.zeros((len(pvals), repeat))
    for i, p in enumerate(pvals):
        for j in range(repeat):
            inds = np.random.choice(K.shape[0], p, replace = False)
            Kt = K[:,inds]
            Ktt = Kt[inds,:]
            yj = y[inds]
            alpha = np.linalg.solve(Ktt + lamb/K.shape[0]*np.eye(p), yj)
            yhat = Kt @ alpha
            err[i,j] = np.mean( (yhat - y)**2 )
    return np.mean(err, axis = 1)

def target_fn(X, beta_opt, K):
    spec = sp.special.iv(np.linspace(1,K,K),beta_opt)**0.5
    cos_mat = np.zeros((K, X.shape[0]))
    for k in range(K):
        cos_mat[k,:] = np.cos(k*X)
        if k ==0:
            cos_mat[k,:] *= 0.5
    y = cos_mat.T @ spec
    return y / 10


np.random.seed(0)
P = 2000
X = np.linspace(-math.pi,math.pi, P)

k_max = 6
#y = np.exp(np.cos(k_max*X))

sigma1 = 0.04
sigma2 = 2
sigma3 = 1/np.sqrt(k_max)

K = 100

beta1 = 1/sigma1**2
beta2 = 1/sigma2**2
beta3 = 1/sigma3**2
y = target_fn(X, beta3, K)

K1 = von_mises(np.array([0]),X,beta1)
K2 = von_mises(np.array([0]),X,beta2)
K3 = von_mises(np.array([0]),X,beta3)
#K3 = np.outer(y[int(len(y)/2)],y) + 1e-4*von_mises(np.array([0]), X, beta3)

plt.figure(figsize=(2.2,1.8))
plt.plot(X, K1[0,:], label ='narrow')
plt.plot(X, K2[0,:], label = 'wide')
plt.plot(X, K3[0,:], label = 'optimal')
plt.title('Different Bandwidths')
plt.ylabel(r'$K(\theta)$', fontsize=myaxis_font)
plt.xlabel(r'$\theta$',fontsize=myaxis_font)
plt.ylim([-0.1, 1.7])
plt.legend()
plt.tight_layout()
plt.savefig('wide_vs_narrow_kernel.pdf')
plt.show()



plt.figure(figsize=(2.2,1.8))
K = 400
s1 = spectrum(beta1,K)
s2 = spectrum(beta2,K)
s3 = spectrum(beta3, K)
plt.loglog(np.linspace(1,len(s1),len(s1)), s1/s1[0], label = 'narrow')
plt.loglog(np.linspace(1,len(s1),len(s1)),s2/s2[0], label = 'wide')
plt.loglog(np.linspace(1,len(s1),len(s1)),s3/s3[0], label = 'optimal')
plt.title('Spectra')
plt.ylabel(r'$\lambda_k$', fontsize=myaxis_font)
plt.xlabel(r'$k$',fontsize=myaxis_font)
plt.ylim([1e-10,10])
#plt.legend()
plt.tight_layout()
plt.savefig('wide_vs_narrow_spectra.pdf')
plt.show()


K1 = von_mises(X,X,beta1)
K2 = von_mises(X,X,beta2)
#K3 = von_mises(X,X,beta3)
K3 = np.outer(y,y)
#y = target_fn(X)
#y = np.cos(5*X)
#y = np.sign(np.cos(X))

#my = y.mean()
#y -= my

p = 8
lamb = 1e-8
x_s = 2*math.pi*np.random.random_sample(p) - math.pi
#ys = target_fn(x_s)
#ys = np.cos(5*x_s)
#ys = np.exp(np.cos(k_max*x_s))
ys = target_fn(x_s, beta3, K)
K1s = von_mises(x_s,x_s, beta1)
K2s = von_mises(x_s,x_s, beta2)
K3s = von_mises(x_s,x_s, beta3)
#K3s = np.outer( np.exp(np.cos(k_max*x_s)), np.exp(np.cos(k_max*x_s)) )
alpha1 = np.linalg.solve(K1s + lamb*np.eye(p), ys)
alpha2 = np.linalg.solve(K2s + lamb*np.eye(p),  ys)
alpha3 = np.linalg.solve(K3s + lamb*np.eye(p), ys)
yhat1 = von_mises(X,x_s, beta1) @ alpha1
yhat2 = von_mises(X, x_s, beta2) @ alpha2
yhat3 = von_mises(X, x_s, beta3) @ alpha3
#yhat3 = np.outer(y, np.exp(np.cos(k_max*x_s)) ) @ alpha3
plt.figure(figsize=(2.4,2))
plt.plot(X, yhat1, label = 'narrow')
plt.plot(X, yhat2, label = 'wide')
plt.plot(X, yhat3, label = 'optimal')
plt.plot(X, y, '--', color = 'black')
plt.plot(x_s, ys, 'o', markersize=3, color = 'black')
plt.ylim([-1,3])
plt.title('Bandwidth and Sample Efficiency')
plt.ylabel(r'$y(\theta)$', fontsize=myaxis_font)
plt.xlabel(r'$\theta$', fontsize=myaxis_font)
plt.tight_layout()
plt.savefig('bandwidth_vs_fit.pdf')
plt.show()

"""


s1,v1 = np.linalg.eigh(K1)
indssort = np.argsort(s1)[::-1]
s1 = s1[indssort]
v1 = v1[:,indssort]

K1 = K1/s1[0]
s1 = s1/s1[0]

s2,v2 = np.linalg.eigh(K2)
indssort = np.argsort(s2)[::-1]
s2 = s2[indssort]
v2 = v2[:,indssort]

K2 = K2/s2[0]
s2 = s2/s2[0]

c1 = (v1.T @ y)**2 / np.dot(y,y)
c2 = (v2.T @ y)**2 / np.dot(y,y)


lamb = 2



pvals = np.logspace(0, 3.5, 100)

#pexp = np.linspace(2, 250, 15).astype('int')
pexp = np.logspace(0, 3, 15 ).astype('int')
exp1 = k_expt(pexp, K1, y, lamb * np.trace(K1))
exp2 = k_expt(pexp, K2, y, lamb * np.trace(K2))

eg1 = theory_curve.mode_errs(pvals, s1, c1, lamb* np.trace(K1)).sum(axis = 0)
eg2 = theory_curve.mode_errs(pvals, s2, c2, lamb * np.trace(K2)).sum(axis = 0)

plt.figure(figsize=(2.4,2))
plt.plot(pvals, eg1, color = 'C0', label = 'narrow')
plt.plot(pvals, eg2, color = 'C1', label = 'wide')
plt.plot(pexp, exp1, 'o', markersize=3, color = 'C0')
plt.plot(pexp, exp2, 'o', markersize=3, color = 'C1')
plt.title('Bandwidth and Generalization')
plt.ylabel(r'$E_g$', fontsize=myaxis_font)
plt.xlabel(r'$p$',fontsize=myaxis_font)
plt.tight_layout()
plt.savefig('bandwidth_learning_curves.pdf')
plt.show()

plt.figure(figsize=(2.4,2))
pscale = (1+pvals)**(-2)
plt.loglog(pvals, eg1, color = 'C0', label = 'narrow')
plt.loglog(pvals, eg2, color = 'C1', label = 'wide')
plt.loglog(pexp, exp1, 'o', markersize=3, color = 'C0')
plt.loglog(pexp, exp2, 'o', markersize=3, color = 'C1')
plt.loglog(pvals[50:len(pvals)], pscale[50:len(pvals)]/pscale[-1] * eg2[-1] ,'--',color = 'black', label = r'$p^{-2}$')
plt.loglog(pvals[50:len(pvals)], pscale[50:len(pvals)]/pscale[-1] * eg1[-1] ,'--',color = 'black')
plt.title('Bandwidth and Generalization')
plt.ylabel(r'$E_g$', fontsize=myaxis_font)
plt.xlabel(r'$p$',fontsize=myaxis_font)
plt.legend()
plt.ylim([1e-4, 5])
plt.tight_layout()
plt.savefig('bandwidth_learning_curves.pdf')
plt.show()
"""


K = 10
lamb = 1e-6
#y = target_k(X, K)
#y = np.exp(np.cos(k_max*X))
sigma_vals = np.linspace(0.01, 1, 50)
pvals = np.linspace(1, 15, 500)
Eg = np.zeros(( len(sigma_vals), len(pvals) ))

for i, sigma in enumerate(tqdm(sigma_vals)):
    beta = 1/sigma**2
    K = von_mises(X,X, beta)
    s, v = np.linalg.eigh(K)
    inds = np.argsort(s)[::-1]
    s = s[inds]
    v = v[:,inds]
    coeff = (v.T @ y)**2 / np.dot(y,y)
    Eg[i,:] = theory_curve.mode_errs(pvals, s, coeff, lamb*np.trace(K)).sum(axis = 0)

plt.figure(figsize=(2.2,1.8))
plt.contourf(Eg, levels = 25, cmap= 'rainbow')
plt.plot(np.linspace(0,len(pvals)-1, len(pvals)), np.argmin(Eg, axis = 0), label = r'optimal $\sigma$', color = 'black')
plt.legend()
plt.xticks(np.linspace(0,len(pvals) -1, 3), np.linspace(np.amin(pvals), np.amax(pvals), 3))
plt.yticks(np.linspace(0,len(sigma_vals) -1, 3), np.linspace(np.amin(sigma_vals), np.amax(sigma_vals), 3))
plt.xlabel(r'$p$',fontsize=myaxis_font)
plt.ylabel(r'$\sigma$',fontsize=myaxis_font)
plt.title(r'Generalization',fontsize=myaxis_font)
cbar =plt.colorbar(fraction = 0.1, ticks = [np.amin(Eg), np.amax(Eg)])
cbar.ax.set_yticklabels([r'%0.1f'% np.amin(Eg), r'%0.1f'% np.amax(Eg)])
plt.tight_layout()
plt.savefig('pure_mode_task_bandwidth_contour.pdf')
plt.show()

plt.figure(figsize=(2.2,1.8))
plt.plot(pvals, Eg[0,:], label = 'narrow')
plt.plot(pvals, Eg[-1,:], label = 'wide')
plt.plot(pvals, Eg[len(Eg)//2,:], label = 'optimal')
plt.xlabel(r'$p$',fontsize=myaxis_font)
plt.ylabel(r'$E_g$',fontsize=myaxis_font)
plt.title(r'Learning Curves',fontsize=myaxis_font)
plt.tight_layout()
plt.savefig('theory_learning_curves.pdf')
plt.show()


km = 100
p_teach = 50
b_teach = 2
X_teach = np.random.uniform(-math.pi, math.pi, p_teach)
K_teach = power_law_kernel(X_teach,X_teach,b_teach,km)
K_teach_test = power_law_kernel(X, X_teach,b_teach,km)
y = K_teach_test @ np.linalg.inv(K_teach) @ np.random.standard_normal(p_teach)
plt.plot(X, y)
plt.show()
print(y.shape)
lamb = 1e-6
bvals = np.linspace(0.5, 2, 20)
pvals = np.linspace(1, 20, 100)
Eg = np.zeros(( len(bvals), len(pvals) ))
for i, b in enumerate(tqdm(bvals)):

    K = power_law_kernel(X,X, b, km)
    s, v = np.linalg.eigh(K)
    inds = np.argsort(s)[::-1]
    s = s[inds]
    v = v[:,inds]
    coeff = (v.T @ y)**2 / np.dot(y,y)
    Eg[i,:] = theory_curve.mode_errs(pvals, s, coeff, lamb*np.trace(K)).sum(axis = 0)

plt.figure(figsize=(1.8,1.5))
plt.contourf(Eg, levels = 25, cmap= 'rainbow')
plt.plot(np.linspace(0,len(pvals)-1, len(pvals)), np.argmin(Eg, axis = 0), label = r'optimal $b$', color = 'black')
plt.legend()
plt.xticks(np.linspace(0,len(pvals) -1, 3), np.linspace(np.amin(pvals), np.amax(pvals), 3))
plt.yticks(np.linspace(0,len(bvals) -1, 3), np.linspace(np.amin(bvals), np.amax(bvals), 3))
plt.xlabel(r'$p$',fontsize=myaxis_font)
plt.ylabel(r'$\sigma$',fontsize=myaxis_font)
plt.title(r'$E_g$ Pure Mode',fontsize=myaxis_font)
cbar =plt.colorbar(fraction = 0.1, ticks = [np.amin(Eg), np.amax(Eg)])
cbar.ax.set_yticklabels([r'%0.1f'% np.amin(Eg), r'%0.1f'% np.amax(Eg)])
plt.tight_layout()
plt.savefig('pure_mode_task_powerlaw_contour.pdf')
plt.show()
