import numpy as np
import numpy.polynomial
import math
import matplotlib.pyplot as plt
import scipy as sp
import scipy.special

import matplotlib as mpl
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

fig_dir = 'figures/'
poly_d = [0,1,2,3]


myfont = 6
myaxis_font = 8
plt.rcParams.update({'font.size': myfont})

ivals = np.linspace(-1,1,num = 100)
plt.figure(figsize=(2.4,2))

"""
q_a_vals = [[0,0], [1,0], [2,0], [1, 0.5], [1, 0.75]]
for q in []:
    if q == 0:
        plt.plot(ivals, np.heaviside(ivals-a, 0), label = r'$q = %d$' % q, linewidth=1)
    else:
        plt.plot(ivals, np.maximum(ivals-a, np.zeros(len(ivals)))**q , label = r'$q = %d$' % q ,linewidth=1)

for a in [0.5,0.75]:
    plt.plot(ivals, np.maximum(ivals-a, np.zeros(len(ivals))))
plt.xlabel(r'$z$', fontsize=myaxis_font)
plt.ylabel(r'$\psi(z)$',fontsize=myaxis_font)
plt.title('Nonlinearities',fontsize=myaxis_font)
plt.legend()
#plt.axis('off')
#plt.xticks([])
#plt.yticks([])
plt.tight_layout()
plt.savefig('f_i_curve_no_legend.pdf')
plt.show()


for i, pd in enumerate(poly_d):
    if pd == 0:
        plt.plot(ivals, np.heaviside(ivals, 0), label = r'$q = %d$' % pd, linewidth=1)
    else:
        plt.plot(ivals, np.maximum(ivals, np.zeros(len(ivals)))**pd , label = r'$q = %d$' % pd ,linewidth=1)
plt.xlabel(r'$z$', fontsize=myaxis_font)
plt.ylabel(r'$\psi(z)$',fontsize=myaxis_font)
plt.title('Nonlinearities',fontsize=myaxis_font)
plt.legend()
#plt.axis('off')
#plt.xticks([])
#plt.yticks([])
plt.tight_layout()
plt.savefig('f_i_curve_no_legend.pdf')
plt.show()


plt.figure(figsize=(1.8,1.5))
plt.plot(ivals, ivals**2, linewidth=4)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig('f_i_curve_complex_no_legend.pdf')
plt.show()

"""



x = np.linspace(-1,1,250)
y = np.linspace(-1,1,250)
lamb = 0.5

N = 35
all_gabor = []
np.random.seed(100)

for n in range(N):
    #phi = math.pi/3
    theta = 2*math.pi* np.random.random_sample()
    phi = 2*math.pi*np.random.random_sample()
    #k = 10 * np.array([np.cos(theta), np.sin(theta)])
    k = 10 * np.random.standard_normal(2)
    gabor = np.zeros((len(x), len(y)))
    for i,xi in enumerate(x):
        for j,yj in enumerate(y):
            v = np.array([xi,yj])
            gabor[i,j] = np.exp(- 1/lamb**2 * np.sqrt(xi**2 + yj**2)**2 ) * np.cos( np.dot(v,k) - phi)
    all_gabor += [gabor]

axs =[plt.subplot(5,7,i+1) for i in range(N)]
for i, a in enumerate(axs):
    a.imshow(all_gabor[i], cmap = 'gray')
    a.axis('off')
plt.tight_layout()
plt.subplots_adjust(wspace = -0.1,hspace=-0.1)
plt.savefig(fig_dir + 'gabor_bank.pdf')
plt.show()

k = 5*np.ones(2)
phi = math.pi
stimulus = np.zeros((len(x),len(y)))
for i,xi in enumerate(x):
    for j,yj in enumerate(y):
        v = np.array([xi,yj])
        stimulus[i,j] = np.cos(np.dot(v,k)+phi)

r_plot = stimulus.shape[0] * 0.4 * np.linspace(0, 1, 100)

plt.figure(figsize=(1.8,1.5))
plt.imshow(stimulus, cmap = 'gray')
plt.plot( stimulus.shape[0]/2 * np.ones(len(r_plot)), stimulus.shape[0]/2 + r_plot, color = 'blue')
plt.plot(stimulus.shape[0]/2 + r_plot*np.sin(math.pi/4), stimulus.shape[0]/2 + r_plot*np.cos(math.pi/4), color = 'blue')
plt.text(stimulus.shape[0]/2 * (1 + 0.05), stimulus.shape[0]/2 * (1+0.6), r'$\theta$', color = 'blue', fontsize=1.75*myaxis_font)
plt.plot(stimulus.shape[0]/2 - phi/(2*math.pi) * r_plot*np.cos(math.pi/4), stimulus.shape[0]/2 - phi/(2*math.pi) * r_plot*np.sin(math.pi/4), color = 'red')
plt.text(stimulus.shape[0]/2 * (1 - 0.25), stimulus.shape[0]/2 * (1-0.4), r'$\phi$', color = 'red', fontsize=1.75*myaxis_font)
plt.axis('off')
plt.title(r'Grating Stimulus',fontsize=8)
plt.tight_layout()
plt.savefig(fig_dir + 'grating_stimulus.pdf')
plt.show()



plt.imshow(gabor, cmap = 'gray')
plt.axis('off')
plt.title(r'$(\theta_i,\phi_i)$',fontsize=40)
plt.tight_layout()
plt.savefig(fig_dir + 'single_gabor.pdf')
plt.show()

phi += math.pi/2
gabor = np.zeros((len(x), len(y)))
for i,xi in enumerate(x):
    for j,yj in enumerate(y):
        v = np.array([xi,yj])
        gabor[i,j] = np.exp(- 1/lamb**2 * np.sqrt(xi**2 + yj**2)**2 ) * np.cos( np.dot(v,k) - phi)


plt.imshow(gabor, cmap = 'gray')
plt.axis('off')
plt.title(r'$(\theta_i,\phi_i+\pi/2)$',fontsize=40)
plt.tight_layout()
plt.savefig(fig_dir + 'single_gabor_phase_shift.pdf')
plt.show()

"""
phi = np.linspace(-math.pi,math.pi,100)
theta = np.linspace(-math.pi,math.pi,100)
R = np.zeros((phi.shape[0],theta.shape[0]))
phi_i = 0
for i in range(phi.shape[0]):
    for j in range(theta.shape[0]):
        R[i,j] = np.cos(phi[i]-phi_i) * np.exp(2*np.cos(theta[j])) + np.cos(phi[i]+phi_i) * np.exp(-2*np.cos(theta[j]))
plt.imshow(R)
plt.xticks([])
plt.yticks([])
plt.xlabel(r'$\theta - \theta_i$',fontsize=20)
plt.ylabel(r'$\phi$',fontsize=20)
plt.title('Simple Cell Tuning Curve', fontsize=20)
plt.tight_layout()
plt.savefig('simple_cells_phase_tuning_curve.pdf')
plt.show()

phi = np.linspace(-math.pi,math.pi,100)
theta = np.linspace(-math.pi,math.pi,100)
R = np.zeros((phi.shape[0],theta.shape[0]))
phi_i = 0

for i in range(phi.shape[0]):
    for j in range(theta.shape[0]):
        R[i,j] = np.exp(2*np.cos(theta[j])) + np.exp(-2*np.cos(theta[j]))
plt.imshow(R)
plt.xticks([])
plt.yticks([])
plt.xlabel(r'$\theta-\theta_i$',fontsize=20)
plt.ylabel(r'$\phi$',fontsize=20)
plt.title('Complex Cell Tuning Curve',fontsize=20)
plt.tight_layout()
plt.savefig('complex_cells_phase_tuning_curve.pdf')
plt.show()


for i in range(phi.shape[0]):
    for j in range(theta.shape[0]):
        R[i,j] = np.cos(phi[i]-phi_i) * np.exp(-2*np.cos(theta[j])) + np.cos(phi[i]+phi_i) * np.exp(2*np.cos(theta[j]))
plt.imshow(R)
plt.xticks(int(100/5)*np.linspace(0,4,5), [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
plt.yticks(int(100/5)*np.linspace(0,4,5), [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
plt.xlabel(r'$\theta$',fontsize=20)
plt.ylabel(r'$\phi$',fontsize=20)
#plt.tight_layout()
plt.savefig('simple_cells_phase_kernel.pdf')
plt.show()



theta = np.linspace(-math.pi, math.pi, 1000)
betavals = [1,2,5,10]
plt.figure(figsize=(1.8,1.5))
for i, beta in enumerate(betavals):
    K = 1/np.cosh(beta) * np.cosh( beta * np.cos(theta) )
    plt.plot(theta, K, label = r'$\beta = %d$' % beta)
plt.xlabel(r'$\Delta \theta$', fontsize=myaxis_font)
plt.ylabel(r'$K(\Delta \theta)$',fontsize=myaxis_font)
plt.legend()
plt.tight_layout()
plt.savefig('kernel_gabor_linear.pdf')
plt.show()

K = 30
eigs = np.zeros((len(betavals),K))
for i,beta in enumerate(betavals):
    for j,k in enumerate(range(K)):
        eigs[i,j] = sp.special.iv(k,beta) + sp.special.iv(k,-beta)

for i, beta in enumerate(betavals):
    ei=[e for e in eigs[i,:] if e > 1e-100]
    plt.semilogy(ei,label = r'$\beta = %d$' % beta)
plt.xlabel(r'$n/2$', fontsize=20)
plt.ylabel(r'$\lambda_n$',fontsize=20)
plt.legend()
plt.tight_layout()
plt.savefig('spectrum_gabor_linear.pdf')
plt.show()




deg = 1500
p, w = np.polynomial.chebyshev.chebgauss(deg)

a = 1.2
func = np.cosh(a * p)
func = func - 0.75*np.cosh(a)

K = 1000
poly_d = [0,1,2,3,4]
coeffs= np.zeros((len(poly_d), K))
for i,pdi in enumerate(poly_d):
    psi = np.maximum(np.zeros(len(p)), func)**pdi
    if pdi == 0:
        psi = np.heaviside(func, np.zeros(len(p)))
    for k in range(K):
        Tk = sp.special.eval_chebyt(k, p)
        c = np.dot(w,psi*Tk)
        if k > 0:
            if np.abs(c) < 0.05*np.sqrt(coeffs[i,k-1]) and k > 0:
                c = 0
        coeffs[i,k] = c**2

plt.figure(figsize=(1.8,1.5))

for i,pd in enumerate(poly_d):
    ci = [c for c in coeffs[i,:] if c > 0]
    plt.loglog(ci, 'o', markersize = 1, label = r'$q = %d$' % pd)
    inds = np.linspace(1,len(ci),len(ci))
    plt.loglog(0.1*inds**(-2*pd-2), '--', color = 'black')
plt.xlabel(r'$n$', fontsize=20)
plt.ylabel(r'$\lambda_n$',fontsize=20)
plt.legend()
plt.tight_layout()
plt.savefig('spectrum_gabor_nonlinear.pdf')
plt.show()

all_cos = np.zeros((K,len(theta)))
for i in range(K):
    all_cos[i,:] = np.cos(i*theta)

K = coeffs @ all_cos
for i,pd in enumerate(poly_d):
    plt.plot(theta,K[i,:]/K[i,0], label = r'$q = %d$' % pd)
plt.xlabel(r'$\theta$', fontsize=20)
plt.ylabel(r'$K(\theta)$',fontsize=20)
plt.legend()
plt.tight_layout()
plt.savefig('kernel_nonlinear.pdf')
plt.show()
"""
