import numpy as np
import matplotlib.pyplot as plt
import math
import scipy as sp
import scipy.special

np.random.seed(0)

def tuning_curve(theta, theta_i, phi, phi_i,sigma, relu = False):
    wh = 0.5*np.cos(phi+phi_i) * np.exp(-sigma*np.cos(theta-theta_i)) + 0.5*np.cos(phi-phi_i) * np.exp(sigma*np.cos(theta-theta_i))
    if relu:
        wh = (wh>0)*wh
    return wh

def kernel(theta, phi,phip,theta_grid,phi_grid):
    N = len(phi_grid)*len(theta_grid)
    sigma = 0.5
    N = 1000
    phi_theta_samples = np.random.random_sample((N,2))
    r = tuning_curve(theta, phi_theta_samples[:,0], phi, phi_theta_samples[:,1], sigma)
    rp = tuning_curve(0, phi_theta_samples[:,0],   phip,  phi_theta_samples[:,1], sigma)
    K = 1/N * r.dot(rp)
    return K



myfont = 15
myaxis_font = 30
plt.rcParams.update({'font.size': myfont})

num_phi = 50
phi_vals = np.linspace(-math.pi,math.pi, 2*num_phi)
deg = 50
p, w = np.polynomial.chebyshev.chebgauss(deg)
theta_vals = p



for k in range(3):
    plt.plot(phi_vals, np.cos((k+1)*phi_vals), label = r'$\cos(%d \theta)$' % (k+1))
    plt.plot(phi_vals, np.sin((k+1)*phi_vals), label = r'$\sin(%d \theta)$' % (k+1))
plt.xlabel(r'$\theta$',fontsize=20)
plt.ylabel(r'$\cos(k \theta), \sin(k\theta)$',fontsize=20)
plt.legend()
plt.tight_layout()
plt.savefig('simple_cells_theta_eigenfunctions.pdf')
plt.show()


T = np.zeros((deg,len(p)))
for k in range(deg):
    T[k,:] = sp.special.eval_chebyt(k, theta_vals)

print("starting decomposition")
theta = math.pi/2
K = np.zeros((theta_vals.shape[0], phi_vals.shape[0],phi_vals.shape[0]))
F = np.zeros((deg, phi_vals.shape[0],phi_vals.shape[0]))
for i,phi_i in enumerate(phi_vals):
    for j,phi_j in enumerate(phi_vals):
        for k,theta_k in enumerate(theta_vals):
            K[k,i,j] = kernel(theta_k ,phi_i,phi_j, theta_vals,phi_vals)
        F[:,i,j] = np.dot(T, K[:,i,j]*w)


all_sk = []
all_vk = []
for k in range(20):
    Fk = F[k,:,:]
    sk,vk = np.linalg.eigh(Fk)
    indk = np.argsort(sk)[::-1]
    sk = sk[indk]
    vk = vk[:,indk]
    for i in range(20):
        all_sk.append(sk[i])
        all_vk.append( vk[:,i] )

ind_sort = np.argsort(all_sk)[::-1]
all_sk = [all_sk[i] for i in ind_sort]
all_vk = [all_vk[i] for i in ind_sort]
for i in range(6):
    plt.plot(phi_vals, all_vk[i], label = r'$v_%d$' % (i+1))

plt.xlabel(r'$\phi$',fontsize=20)
plt.ylabel(r'$v_k(\phi)$',fontsize=20)
plt.legend()
plt.tight_layout()
plt.savefig('simple_cells_phase_eigenfunctions.pdf')
plt.show()

power = np.zeros(len(all_vk))
for i,vi in enumerate(all_vk):
    power[i] = np.dot(vi, np.ones(len(vi)))**2

c = np.cumsum(power) / np.sum(power)
plt.plot(c, label = 'simple cells')
plt.plot(np.ones(len(c)), label ='complex cells')
plt.xlabel(r'$k$',fontsize=20)
plt.ylabel(r'$C(k)$',fontsize=20)
plt.legend()
plt.tight_layout()
plt.savefig('cumulative_power_gabor_simple_vs_complex.pdf')
plt.show()


plt.plot(phi_vals, np.ones(len(phi_vals)), label = r'$v_0$')
plt.xlabel(r'$\phi$',fontsize=20)
plt.ylabel(r'$v_k(\phi)$',fontsize=20)
plt.legend()
plt.tight_layout()
plt.savefig('complex_cells_phase_eigenfunctions.pdf')
plt.show()

plt.semilogy(all_sk)
plt.xlabel(r'$k$',fontsize=20)
plt.ylabel(r'$\lambda_k$',fontsize=20)
plt.tight_layout()
plt.savefig('simple_cells_phase_spectrum.pdf')
plt.show()



# fourier decomposition on the

plt.imshow(K[0,:,:])
plt.xticks([])
plt.yticks([])
plt.xlabel(r'$\phi_1$',fontsize=20)
plt.ylabel(r'$\phi_2$',fontsize=20)
plt.tight_layout()
plt.savefig('simple_cells_phase_gabor.pdf')
plt.show()
