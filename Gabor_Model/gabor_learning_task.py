import numpy as np
import math
import matplotlib.pyplot as plt
import power_law
import matplotlib as mpl
from cycler import cycler
from tqdm import tqdm

mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')


def gabor_features_no_phase(theta_stim, theta_pr, sigma, thresh, a = 1.3):
    th_diff = np.outer(theta_stim, np.ones(len(theta_pr))) - np.outer(np.ones(len(theta_stim)), theta_pr)
    cos_th = np.cos(th_diff)
    R = np.cosh(sigma*cos_th)/ np.cosh(sigma)
    R = np.maximum(R-thresh,0)**a
    return 1/np.sqrt(N) * R.T

def sort_svd(R):
    u,s,v = np.linalg.svd(R)
    inds = np.argsort(s)[::-1]
    s=s[inds]**2
    v = v[inds,:].T
    return u,s,v

def sort_eigh(K):
    s,v = np.linalg.eigh(K)
    inds = np.argsort(s)[::-1]
    return s[inds], v[:,inds]

def kernel_regression_expt(K, y, lamb, pvals, num_rep = 30):
    N = K.shape[0]
    errs = np.zeros((len(pvals), num_rep))
    for i, p in enumerate(pvals):
        for j in range(num_rep):
            inds = np.random.choice(N, p)
            k_tr_test = K[:,inds]
            k_tr_tr = k_tr_test[inds,:]
            y_tr = y[inds]
            alpha = np.linalg.solve(k_tr_tr + 1/N*lamb*np.eye(p), y_tr)
            yhat = k_tr_test @ alpha
            errs[i,j] = np.mean( (yhat - y)**2 )
    return np.mean(errs, axis = 1)


# compute the response of simple cell population for (theta,phi) pairs
# data is P x 2
def gabor_simple_cells(data, params, sigma, thresh = 0.1, a = 1.3):
    N = params.shape[0]
    P = data.shape[0]
    theta = data[:,0]
    phi = data[:,1]
    theta_pr = params[:,0]
    phi_pr = params[:,1]

    # N x P matrix
    diff_th = np.outer( theta_pr, np.ones(P) ) - np.outer(np.ones(N), theta)
    diff_phi = np.outer( phi_pr, np.ones(P) ) - np.outer(np.ones(N), phi)
    sum_phi = np.outer(  phi_pr, np.ones(P) ) + np.outer(np.ones(N), phi)

    R = 0.5 * np.exp(-sigma*(1+np.cos(diff_th))) * np.cos(sum_phi) + 0.5 * np.exp(-sigma*(1-np.cos(diff_th))) * np.cos(diff_phi)
    m = np.amax(R)
    tm = thresh * m
    if a > 0:
        R = np.maximum(R-tm, 0)**a
    else:
        R = 1.0 * (R > tm)
    return R

def gabor_complex(data, params, sigma):
    N = params.shape[0]
    P = data.shape[0]
    theta = data[:,0]
    phi = data[:,1]
    theta_pr = params[:,0]
    phi_pr = params[:,1]
    diff_th = np.outer( theta_pr, np.ones(P)) - np.outer(np.ones(N), theta)
    #R = 0.5 *np.cosh(2*sigma*np.cos(diff_th))/np.cosh(2*sigma) + 0.5 * np.outer(np.cos(2*phi_pr),np.ones(P))
    R =  0.5 *np.cosh(2*sigma*np.cos(diff_th))/np.cosh(2*sigma)
    return R

def target_fn_phase(data, k_ori =2, k_ph = 0):

    theta = data[:,0]
    phi = data[:,1]
    return np.sign( np.cos(k_ori*theta) * np.cos(k_ph*phi) )

def target_slant(data):
    theta = data[:,0]
    phi = data[:,1]
    return np.sign( (phi - 2*theta)**2 - 1 )

def target_oval(data):
    theta = data[:,0]
    phi = data[:,1]
    a = math.pi/4
    b = 3*a
    return np.sign(-theta**2/a**2 + phi**2/b**2 + 1)

def target_fn_no_phase(theta, kori = 2):
    return np.sign(np.cos(kori*theta))

def partial_phase_selective_population(data, params, sigma,thresh, a = 1.3):
    N = params.shape[0]
    P = data.shape[0]
    theta = data[:,0]
    phi = data[:,1]
    theta_pr = params[:,0]
    phi_pr = params[:,1]
    shifts = params[:,2]
    #print(shifts)
    # N x P matrix
    diff_th = np.outer( theta_pr, np.ones(P) ) - np.outer(np.ones(N), theta)
    diff_phi = np.outer( phi_pr, np.ones(P) ) - np.outer(np.ones(N), phi)
    sum_phi = np.outer(  phi_pr, np.ones(P) ) + np.outer(np.ones(N), phi)

    R_unshift = 0.5 * np.exp(-sigma*(1+np.cos(diff_th))) * np.cos(sum_phi) + 0.5 * np.exp(-sigma*(1-np.cos(diff_th))) * np.cos(diff_phi)
    R_shift = 0.5 * np.exp(-sigma*(1+np.cos(diff_th))) * np.cos(sum_phi+shifts[:,np.newaxis]) + 0.5 * np.exp(-sigma*(1-np.cos(diff_th))) * np.cos(diff_phi-shifts[:,np.newaxis])

    #Current = R_unshift**2 + R_shift**2 - thresh**2
    Current = np.sqrt(R_unshift**2 + R_shift**2) - thresh
    Current = Current * (Current > 0.0)
    R = Current**(a)
    return R

def dirichlet_mixture_code(data, params, sigma,thresh, a = 1.3):
    N = params.shape[0]
    P = data.shape[0]
    theta = data[:,0]
    phi = data[:,1]
    theta_pr = params[:,0]
    phi_pr = params[:,1]
    betas = params[:,2:]
    #print(shifts)
    # N x P matrix
    diff_th = np.outer( theta_pr, np.ones(P) ) - np.outer(np.ones(N), theta)
    diff_phi = np.outer( phi_pr, np.ones(P) ) - np.outer(np.ones(N), phi)
    sum_phi = np.outer(  phi_pr, np.ones(P) ) + np.outer(np.ones(N), phi)

    R1 = np.exp(sigma*np.cos(diff_th)) * np.cos(diff_phi)
    R2 = np.exp(sigma*np.cos(diff_th)) * np.cos(diff_phi+np.pi/2.0)
    R3 = np.exp(sigma*np.cos(diff_th)) * np.cos(diff_phi+np.pi)
    R4 = np.exp(sigma*np.cos(diff_th)) * np.cos(diff_phi+3.0*np.pi/2.0)

    b1 = np.outer(betas[:,0], np.ones(R1.shape[1]))
    b2 = np.outer(betas[:,1], np.ones(R1.shape[1]))
    b3 = np.outer(betas[:,2], np.ones(R1.shape[1]))
    b4 = np.outer(betas[:,3], np.ones(R1.shape[1]))

    Rsq = b1 * R1**2 * (R1 > 0.0) +b2*R2**2 * (R2 > 0.0) + b3*R3**2 * (R3 > 0.0)+b4* R4**2 * (R4 > 0.0)

    #R_unshift = 0.5 * np.exp(-sigma*(1+np.cos(diff_th))) * np.cos(sum_phi) + 0.5 * np.exp(-sigma*(1-np.cos(diff_th))) * np.cos(diff_phi)
    #R_shift = 0.5 * np.exp(-sigma*(1+np.cos(diff_th))) * np.cos(sum_phi+shifts[:,np.newaxis]) + 0.5 * np.exp(-sigma*(1-np.cos(diff_th))) * np.cos(diff_phi-shifts[:,np.newaxis])
    #Current = R_unshift**2 + R_shift**2 - thresh**2
    Current = np.sqrt(Rsq) - thresh
    Current = Current * (Current > 0.0)
    R = Current**(a)
    return R

# task 1: y = sign( cos theta )
# task 2: y = sign( cos theta cos phi)

myfont = 6
myaxis_font = 8
plt.rcParams.update({'font.size': myfont})
line_width = 1

def plot_tool(x_list, y_list, leg_labels, xlabel, ylabel, title, file_name, x_expt = [], y_expt=[], style = 'loglog', move_legend = False):
    plt.figure(figsize=(2.4,2))
    #plt.subplot(211)
    if style == 'loglog':
        for i in range(len(x_list)):
            plt.loglog(x_list[i], y_list[i], label = leg_labels[i], linewidth=line_width)
            if len(x_expt) > 0:
                plt.loglog(x_expt[i], y_expt[i], 'o', markersize=1, color = 'C%d' %i, linewidth=line_width)
    elif style == 'linear':
        for i in range(len(x_list)):
            plt.plot(x_list[i], y_list[i], label = leg_labels[i], linewidth=line_width)
            if len(x_expt) > 0:
                plt.plot(x_expt[i], y_expt[i], 'o', markersize=1, color = 'C%d' %i, linewidth=line_width)

    elif style == 'semilogx':
        for i in range(len(x_list)):
            plt.semilogx(x_list[i], y_list[i], label = leg_labels[i], linewidth=line_width)
            if len(x_expt) > 0:
                plt.semilogx(x_expt[i], y_expt[i], 'o', markersize=1, color = 'C%d' %i, linewidth=line_width)

    plt.xlabel(xlabel, fontsize=myaxis_font)
    plt.ylabel(ylabel, fontsize=myaxis_font)
    plt.title(title, fontsize=myaxis_font)
    #plt.legend(bbox_to_anchor=(1.8,1), loc = 'upper left', ncol = 1)
    #ax.legend(bbox_to_anchor=(1.1, 1.05))
    if move_legend == False:
        plt.legend()
    else:
        plt.legend(bbox_to_anchor=(1,1), loc = 'upper left', ncol = 1)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()
    return


# compute the F1 power and the F0 power for the code
def compute_F1_F0(R, data):

    N = params.shape[0]
    P = data.shape[0]
    theta = data[:,0]
    phi = data[:,1]
    cos_fn = np.cos(phi)
    sin_fn = np.sin(phi)

    proj_cos = 1/ P * R @ cos_fn
    proj_sin = 1/P * R @ sin_fn
    F1 = np.sqrt(proj_cos**2 + proj_sin**2) # R^N
    F0 = 1/P * R @ np.ones(P) # R^N
    return F1 / F0


np.random.seed(0)

N = 8000
P = 2000

# simple cell parameters from fit
thresh = 0.25
sigma = 5
q = 2.5
#lamb = 1.5
lamb = 2.5
num_theta = 50
num_phi = 50
theta_vals = np.linspace(-math.pi/2,math.pi/2, num_theta)
phi_vals = np.linspace(-math.pi,math.pi, num_phi)
data = np.zeros((num_theta,num_phi,2))
for i in range(num_theta):
    for j in range(num_phi):
        data[i,j,0] = theta_vals[i]
        data[i,j,1] = phi_vals[j]


data_flat = data.reshape((num_theta * num_phi, 2))
params = 2*math.pi*np.random.random_sample((N,6))
R_comp = gabor_complex(data_flat, params, sigma)
R_simp = gabor_simple_cells(data_flat, params, sigma)
print("got simple and complex codes")
"""
plt.figure(figsize=(2.4,2))
plt.imshow(R_comp[1,:].reshape((num_theta,num_phi)).transpose())
#plt.xticks([])
#plt.yticks([])
plt.xticks([0,num_phi-1], [r'$0$', r'$\pi$'])
plt.yticks([0,num_theta-1], [r'$2\pi$', r'$0$'])
plt.xlabel(r'$\theta$',fontsize=myaxis_font)
plt.ylabel(r'$\phi$',fontsize=myaxis_font)
plt.title('Example Complex Cell', fontsize=myaxis_font)
plt.tight_layout()
plt.savefig('complex_cell_tuning_curve_plot.pdf')
plt.show()

plt.figure(figsize=(2.4,2))
plt.imshow(R_simp[2,:].reshape((num_theta,num_phi)).transpose())
plt.xticks([0,num_phi-1], [r'$0$', r'$\pi$'])
plt.yticks([0,num_theta-1], [r'$2\pi$', r'$0$'])
plt.xlabel(r'$\theta$',fontsize=myaxis_font)
plt.ylabel(r'$\phi$',fontsize=myaxis_font)
plt.title('Example Simple Cell', fontsize=myaxis_font)
plt.tight_layout()
plt.savefig('simple_cell_tuning_curve_plot.pdf')
plt.show()
"""


K_comp = 1/R_comp.shape[0] * R_comp.T @ R_comp
K_simp = 1/R_simp.shape[0] * R_simp.T @ R_simp


K_comp = K_comp / np.trace(K_comp)
K_simp = K_simp / np.trace(K_simp)

s, v = sort_eigh(K_comp)
plt.figure(figsize=(1.8,1.5))
for i in range(12):
    plt.subplot(3,4,i+1)
    plt.imshow(v[:,i].reshape((50,50)).T )
    plt.xticks([])
    plt.yticks([])
plt.suptitle(r'Complex Code $\psi_k(\theta,\phi)$',fontsize=myaxis_font)
plt.savefig('figures/complex_pop_eigenfunctions.pdf')
plt.show()

"""
plt.figure(figsize=(1.8,1.5))
for i in range(12):
    plt.subplot(3,4,i+1)
    plt.imshow(R_comp[i,:].reshape((50,50)).T )
    plt.xticks([])
    plt.yticks([])
plt.suptitle(r'Complex Code $r_i(\theta,\phi)$',fontsize=myaxis_font)
plt.savefig('figures/complex_tuning_curves.pdf')
plt.show()
"""

s, v = sort_eigh(K_simp)
plt.figure(figsize=(1.8,1.5))
for i in range(12):
    plt.subplot(3,4,i+1)
    plt.imshow(v[:,i].reshape((50,50)).T)
    plt.xticks([])
    plt.yticks([])
plt.suptitle(r'Simple Code $\psi_k(\theta,\phi)$',fontsize=myaxis_font)
plt.savefig('figures/simple_pop_eigenfunctions.pdf')
plt.show()

K_mix = K_simp * 0.4 + K_comp * 0.6
s, v = sort_eigh(K_mix)
plt.figure(figsize=(1.8,1.5))
for i in range(12):
    plt.subplot(3,4,i+1)
    plt.imshow(v[:,i].reshape((50,50)).T)
    plt.xticks([])
    plt.yticks([])
plt.suptitle(r'Mixture Code $\psi_k(\theta,\phi)$',fontsize=myaxis_font)
plt.savefig('figures/mixture_pop_eigenfunctions.pdf')
plt.show()

"""
plt.figure(figsize=(1.8,1.5))
for i in range(12):
    plt.subplot(3,4,i+1)
    plt.imshow(R_simp[i,:].reshape((50,50)).T )
    plt.xticks([])
    plt.yticks([])
plt.suptitle(r'Simple Code $r_i(\theta,\phi)$',fontsize=myaxis_font)
plt.savefig('figures/simple_tuning_curves.pdf')
plt.show()
"""




# set up learning tasks

y_ori   = target_fn_phase(data_flat, k_ori = 2, k_ph = 0)
y_phase = target_fn_phase(data_flat, k_ori = 0, k_ph = 1)
y_slant = target_oval(data_flat)
#y_both = target_fn_phase(data_flat, k_ori = 2, k_ph=1)

"""
plt.figure(figsize=(1.8,1.5))
plt.imshow(y_ori.reshape((num_theta,num_phi)).transpose())
plt.xticks([0,num_phi-1], [r'$0$', r'$\pi$'])
plt.yticks([0,num_theta-1], [r'$2\pi$', r'$0$'])
plt.xlabel(r'$\theta$',fontsize=myaxis_font)
plt.ylabel(r'$\phi$',fontsize=myaxis_font)
plt.title('Orientation Task', fontsize=myaxis_font)
plt.tight_layout()
plt.savefig('ori_task_visual.pdf')
plt.show()

plt.figure(figsize=(1.8,1.5))
plt.imshow(y_phase.reshape((num_theta,num_phi)).transpose())
plt.xticks([0,num_phi-1], [r'$0$', r'$\pi$'])
plt.yticks([0,num_theta-1], [r'$2\pi$', r'$0$'])
plt.xlabel(r'$\theta$',fontsize=myaxis_font)
plt.ylabel(r'$\phi$',fontsize=myaxis_font)
plt.title('Phase Task', fontsize=myaxis_font)
plt.tight_layout()
plt.savefig('phase_task_visual.pdf')
plt.show()


plt.figure(figsize=(1.8,1.5))
plt.imshow(y_slant.reshape((num_theta,num_phi)).transpose())
plt.xticks([0,num_phi-1], [r'$0$', r'$\pi$'])
plt.yticks([0,num_theta-1], [r'$2\pi$', r'$0$'])
plt.xlabel(r'$\theta$',fontsize=myaxis_font)
plt.ylabel(r'$\phi$',fontsize=myaxis_font)
plt.title('Hybrid Task', fontsize=myaxis_font)
plt.tight_layout()
plt.savefig('slant_task_visual.pdf')
plt.show()
"""



alpha_vals = np.logspace(-2,4.0,num = 30)
#alpha_vals = np.linspace(0.0,np.pi/2, num= 30)
ptheory = np.logspace(0, 2.0, 80)


gen_err_ori = np.zeros((len(alpha_vals), len(ptheory)))
gen_err_phase = np.zeros((len(alpha_vals), len(ptheory)))
gen_err_slant = np.zeros((len(alpha_vals), len(ptheory)))

cum_ori = np.zeros((len(alpha_vals), K_simp.shape[0]))
cum_phase= np.zeros((len(alpha_vals), K_simp.shape[0]))
cum_slant = np.zeros((len(alpha_vals), K_simp.shape[0]))


for i,alpha in enumerate(tqdm(alpha_vals)):
    np.random.seed(i)
    #betas = np.random.gamma(alpha, size = (N,4))
    #betas = betas / betas.sum(axis = 1)[:,np.newaxis]
    betas = np.random.dirichlet(alpha*np.ones(4), size = N)
    print(betas[0:5,:])
    print(np.mean(np.std(betas, axis = 1)))
    params[:,2:] = betas
    #params[:,2] = np.pi/2.0 + 1/alpha * np.random.standard_normal(N)
    #params[:,2] = 1/alpha * np.random.standard_normal(N)
    #params[:,2] = np.pi/2.0 + alpha  + 1/5.0*np.random.standard_normal(N)
    #R = partial_phase_selective_population(data_flat, params, sigma, thresh)
    R = dirichlet_mixture_code(data_flat, params, sigma,thresh)
    K = 1/R.shape[0] * R.T @ R
    F1F0 = compute_F1_F0(R, data_flat)

    #K = t*K_simp + (1-t)*K_comp
    s, v = sort_eigh(K)
    K = K/s[0]
    s = s/s[0]
    if i == 0:
        v_low = v
        R_low = R
        F1F0_low = F1F0
    #lamb_t = lamb * np.trace(K)
    lamb_t = lamb
    # task 1: only orientation
    coeff_ori = 1/(K.shape[0]) * (v.T @ y_ori)**2
    gen_err_ori[i,:] = power_law.mode_errs(ptheory, s, coeff_ori, lamb_t).sum(axis = 0)
    cum_ori[i,:] = np.cumsum(coeff_ori) / np.sum(coeff_ori) # check to see if this is backwards  (left to right is ideal)

    # task 2: only phase
    coeff_phase =  1/(K.shape[0]) *(v.T @ y_phase)**2
    gen_err_phase[i,:] = power_law.mode_errs(ptheory, s, coeff_phase, lamb_t).sum(axis = 0)
    cum_phase[i,:] = np.cumsum(coeff_phase) / np.sum(coeff_phase)

    # task 3: hybrid
    coeff_slant =  1/(K.shape[0]) *(v.T @ y_slant)**2
    gen_err_slant[i,:] = power_law.mode_errs(ptheory, s, coeff_slant, lamb_t).sum(axis = 0)
    cum_slant[i,:] = np.cumsum(coeff_slant) / np.sum(coeff_slant)




plt.figure(figsize=(1.8,1.5))
plt.hist(F1F0_low, density = True)
plt.xlabel('F1/F0',fontsize=myaxis_font)
plt.ylabel('Density',fontsize=myaxis_font)
plt.xlim([0,1])
plt.tight_layout()
plt.savefig('figures/low_alpha_F1_F0.pdf')
plt.show()


plt.figure(figsize=(1.8,1.5))
plt.hist(F1F0, density = True)
plt.xlabel('F1/F0',fontsize=myaxis_font)
plt.ylabel('Density',fontsize=myaxis_font)
plt.xlim([0,1])
plt.tight_layout()
plt.savefig('figures/high_alpha_F1_F0.pdf')
plt.show()

print(tvals)

plt.figure(figsize=(1.8,1.5))
for i in range(12):
    plt.subplot(4,3,i+1)
    plt.imshow(v_low[:,i].reshape((50,50)))
    plt.xticks([])
    plt.yticks([])
    #plt.title('k=%d' % (i +1),fontsize=myaxis_font)
#plt.tight_layout()
plt.savefig('figures/low_alpha_eigenfunctions.pdf')
plt.show()

plt.figure(figsize=(1.8,1.5))
for i in range(12):
    plt.subplot(4,3,i+1)
    plt.imshow(v_low[:,i].reshape((50,50)))
    plt.xticks([])
    plt.yticks([])
    #plt.title('k=%d' % (i +1),fontsize=myaxis_font)
#plt.tight_layout()
plt.savefig('figures/low_alpha_eigenfunctions.pdf')
plt.show()



plt.figure(figsize=(1.8,1.5))
for i in range(12):
    plt.subplot(4,3,i+1)
    plt.imshow(v[:,i].reshape((50,50)))
    plt.xticks([])
    plt.yticks([])
    #plt.title('k=%d' % (i +1))
plt.savefig('figures/high_alpha_eigenfunctions.pdf')
plt.show()


plt.figure(figsize=(1.8,1.5))
for i in range(12):
    plt.subplot(4,3,i+1)
    plt.imshow(R_low[i,:].reshape((50,50)))
    plt.xticks([])
    plt.yticks([])
    #plt.title('k=%d' % (i +1),fontsize=myaxis_font)
#plt.tight_layout()
plt.savefig('figures/low_alpha_code.pdf')
plt.show()

plt.figure(figsize=(1.8,1.5))
for i in range(12):
    plt.subplot(4,3,i+1)
    plt.imshow(R[i,:].reshape((50,50)))
    plt.xticks([])
    plt.yticks([])
    #plt.title('k=%d' % (i +1),fontsize=myaxis_font)
#plt.tight_layout()
plt.savefig('figures/high_alpha_code.pdf')
plt.show()

min_alpha = np.log10(np.amin(alpha_vals))
max_alpha = np.log10(np.amax(alpha_vals))

plt.figure(figsize=(1.8,1.5))
plt.contourf(gen_err_ori, levels = 25, cmap= 'rainbow')
plt.plot(np.linspace(0,len(ptheory)-1, len(ptheory)), np.argmin(gen_err_ori,axis = 0) - 0.4 , label = r'optimal $\alpha$', color = 'black')
plt.legend()
plt.xticks(np.linspace(0,len(ptheory) -1, 3), [1,50,100])
plt.yticks(np.linspace(0,len(alpha_vals) -1, 2),  np.linspace(min_alpha, max_alpha, 2))
#plt.yticks(np.linspace(0,len(alpha_vals) -1, 2), np.linspace(np.amin(np.log10(alpha_vals)), np.amax(np.log10(alpha_vals)), 2))
plt.ylim([0,len(alpha_vals)-1])
plt.xlabel(r'$P$',fontsize=myaxis_font)
plt.ylabel(r'$\log_{10} \ \alpha$',fontsize=myaxis_font)
plt.title(r'$E_g$ Orientation',fontsize=myaxis_font)
cbar =plt.colorbar(fraction = 0.1, ticks = [np.amin(gen_err_ori), np.amax(gen_err_ori)])
cbar.ax.set_yticklabels([r'%0.1f'% np.amin(gen_err_ori), r'%0.1f'% np.amax(gen_err_ori)])
plt.tight_layout()
plt.savefig('figures/partial_phase_ori_task_contour.pdf')
plt.show()


plt.figure(figsize=(1.8,1.5))
num_k = 100
log_cum = np.log10(1.0 - cum_ori[:,0:num_k])
plt.contourf(log_cum, levels = 25, cmap= 'rainbow')
#plt.legend()
plt.xticks(np.linspace(0,num_k, 3), np.linspace(0, num_k, 3))
plt.yticks(np.linspace(0,len(alpha_vals) -1, 2),  np.linspace(min_alpha, max_alpha, 2))

#plt.yticks(np.linspace(0,len(alpha_vals) -1, 2), np.linspace(np.amin(alpha_vals), np.amax(alpha_vals), 2))
plt.xlabel(r'$k$',fontsize=myaxis_font)
plt.ylabel(r'$\log_{10} \ \alpha$',fontsize=myaxis_font)
plt.title(r'$\log(1-C(k))$ Orientation',fontsize=myaxis_font)
cbar =plt.colorbar()
cbar.set_ticks([np.amin(log_cum),np.amax(log_cum)])
cbar.set_ticklabels([r'%0.1f'% np.amin(log_cum), r'%0.1f'% np.amax(log_cum)])
plt.tight_layout()
plt.savefig('figures/partial_phase_ori_task_CK_contour.pdf')
plt.show()

plt.figure(figsize=(1.8,1.5))
plt.contourf(gen_err_phase, levels = 25, cmap= 'rainbow')
plt.plot(np.linspace(0,len(ptheory)-1, len(ptheory)), np.argmin(gen_err_phase,axis = 0)+0.4, color = 'black')
plt.xticks(np.linspace(0,len(ptheory)-1, 3), [1,50,100])
plt.yticks(np.linspace(0,len(alpha_vals) -1, 2),  np.linspace(min_alpha, max_alpha, 2))

#plt.yticks(np.linspace(0,len(alpha_vals)-1, 2), np.linspace(np.amin(alpha_vals), np.amax(alpha_vals), 2))
plt.ylim([0,len(alpha_vals)-1])
plt.xlabel(r'$P$',fontsize=myaxis_font)
plt.ylabel(r'$\log_{10} \ \alpha$',fontsize=myaxis_font)
plt.ylim([0,len(alpha_vals)-1])
plt.title(r'$E_g$ Phase',fontsize=myaxis_font)
cbar = plt.colorbar(fraction = 0.1, ticks = [np.amin(gen_err_phase), np.amax(gen_err_phase)])
cbar.ax.set_yticklabels([r'%0.1f'% np.amin(gen_err_phase), r'%0.1f'% np.amax(gen_err_phase)])
plt.tight_layout()
plt.savefig('figures/partial_phase_phase_task_contour.pdf')
plt.show()


plt.figure(figsize=(1.8,1.5))
num_k = 100
log_cum = np.log10(1.0-cum_phase[:,0:num_k])
print(log_cum)
plt.contourf(log_cum, levels = 25, cmap= 'rainbow')
#plt.legend()
plt.xticks(np.linspace(0,num_k, 3), np.linspace(0, num_k, 3))
plt.yticks(np.linspace(0,len(alpha_vals) -1, 2),  np.linspace(min_alpha, max_alpha, 2))

#plt.yticks(np.linspace(0,len(alpha_vals) -1, 3), np.linspace(np.amin(alpha_vals), np.amax(alpha_vals), 3))
plt.xlabel(r'$k$',fontsize=myaxis_font)
plt.ylabel(r'$\log_{10} \ \alpha$',fontsize=myaxis_font)

plt.title(r'$\log(1-C(k))$ Phase',fontsize=myaxis_font)
#cbar =plt.colorbar(fraction = 0.1, ticks = [np.amin(cum_ori[0:num_k]), np.amax(cum_ori[0:num_k])])
cbar = plt.colorbar()
cbar.set_ticks([np.amin(log_cum),np.amax(log_cum)])
cbar.set_ticklabels([r'%0.1f'% np.amin(log_cum), r'%0.1f'% np.amax(log_cum)])
plt.tight_layout()
plt.savefig('figures/partial_phase_phase_task_CK_contour.pdf')
plt.show()

plt.figure(figsize=(1.8,1.5))
plt.contourf(gen_err_slant, levels = 25, cmap = 'rainbow')
plt.plot(np.linspace(0,len(ptheory)-1, len(ptheory)), np.argmin(gen_err_slant,axis = 0)-0.4, color = 'black')
plt.xticks(np.linspace(0,len(ptheory)-1, 3), [1,50,100])
plt.yticks(np.linspace(0,len(alpha_vals) -1, 2),  np.linspace(min_alpha, max_alpha, 2))

#plt.yticks(np.linspace(0,len(alpha_vals)-1, 3), np.linspace(np.amin(alpha_vals), np.amax(alpha_vals), 3))
plt.ylim([0,len(alpha_vals)-1])

plt.xlabel(r'$P$',fontsize=myaxis_font)
plt.ylabel(r'$\log_{10} \ \alpha$',fontsize=myaxis_font)
plt.title(r'$E_g$ Hybrid',fontsize=myaxis_font)
cbar = plt.colorbar(fraction = 0.1, ticks = [np.amin(gen_err_slant), np.amax(gen_err_slant)])
cbar.ax.set_yticklabels([r'%0.1f'% np.amin(gen_err_slant), r'%0.1f'% np.amax(gen_err_slant)])
plt.tight_layout()
plt.savefig('figures/partial_phase_hybrid_task_contour.pdf')
plt.show()


plt.figure(figsize=(1.8,1.5))
num_k = 100
log_cum = np.log10(1.0 - cum_slant[:,0:num_k])
plt.contourf(log_cum, levels = 25, cmap= 'rainbow')
#plt.legend()
plt.xticks(np.linspace(0,num_k, 3), np.linspace(0, num_k, 3))
plt.yticks(np.linspace(0,len(alpha_vals) -1, 2),  np.linspace(min_alpha, max_alpha, 2))

#plt.yticks(np.linspace(0,len(alpha_vals) -1, 3), np.linspace(np.amin(alpha_vals), np.amax(alpha_vals), 3))
plt.xlabel(r'$k$',fontsize=myaxis_font)
plt.ylabel(r'$\log_{10} \ \alpha$',fontsize=myaxis_font)
plt.title(r'$\log(1-C(k))$ Hybrid',fontsize=myaxis_font)
cbar =plt.colorbar()
cbar.set_ticks([np.amin(log_cum),np.amax(log_cum)])
cbar.set_ticklabels([r'%0.1f'% np.amin(log_cum), r'%0.1f'% np.amax(log_cum)])
plt.tight_layout()
plt.savefig('figures/partial_phase_hybrid_task_CK_contour.pdf')
plt.show()


plt.plot(tvals, tvals**2)
plt.show()

tvals = np.linspace(0,1,num = 30)
ptheory = np.logspace(0, 3.5, 80)


gen_err_ori = np.zeros((len(tvals), len(ptheory)))
gen_err_phase = np.zeros((len(tvals), len(ptheory)))
gen_err_slant = np.zeros((len(tvals), len(ptheory)))

cum_ori = np.zeros((len(tvals), K_simp.shape[0]))
cum_phase= np.zeros((len(tvals), K_simp.shape[0]))
cum_slant = np.zeros((len(tvals), K_simp.shape[0]))

for i,t in enumerate(tqdm(tvals)):
    K = t*K_simp + (1-t)*K_comp
    s, v = sort_eigh(K)
    K = K/s[0]
    s = s/s[0]

    lamb_t = lamb * np.trace(K)
    # task 1: only orientation
    coeff_ori = 1/(K.shape[0]) * (v.T @ y_ori)**2
    gen_err_ori[i,:] = power_law.mode_errs(ptheory, s, coeff_ori, lamb_t).sum(axis = 0)
    cum_ori[i,:] = np.cumsum(coeff_ori) / np.sum(coeff_ori) # check to see if this is backwards  (left to right is ideal)

    # task 2: only phase
    coeff_phase =  1/(K.shape[0]) *(v.T @ y_phase)**2
    gen_err_phase[i,:] = power_law.mode_errs(ptheory, s, coeff_phase, lamb_t).sum(axis = 0)
    cum_phase[i,:] = np.cumsum(coeff_phase) / np.sum(coeff_phase)

    # task 3: hybrid
    coeff_slant =  1/(K.shape[0]) *(v.T @ y_slant)**2
    gen_err_slant[i,:] = power_law.mode_errs(ptheory, s, coeff_slant, lamb_t).sum(axis = 0)
    cum_slant[i,:] = np.cumsum(coeff_slant) / np.sum(coeff_slant)

plt.figure(figsize=(1.8,1.5))
plt.contourf(gen_err_ori, levels = 25, cmap= 'rainbow')
plt.plot(np.linspace(0,len(ptheory)-1, len(ptheory)), np.argmin(gen_err_ori,axis = 0) + 0.4 , label = r'optimal $s$', color = 'black')
plt.legend()
plt.xticks(np.linspace(0,len(ptheory) -1, 3), [1,50,100])
plt.yticks(np.linspace(0,len(tvals) -1, 3), np.linspace(np.amin(tvals), np.amax(tvals), 3))
plt.xlabel(r'$P$',fontsize=myaxis_font)
plt.ylabel(r'$s$',fontsize=myaxis_font)
plt.title(r'$E_g$ Orientation',fontsize=myaxis_font)
cbar =plt.colorbar(fraction = 0.1, ticks = [np.amin(gen_err_ori), np.amax(gen_err_ori)])
cbar.ax.set_yticklabels([r'%0.1f'% np.amin(gen_err_ori), r'%0.1f'% np.amax(gen_err_ori)])
plt.tight_layout()
plt.savefig('ori_task_contour.pdf')
plt.show()


plt.figure(figsize=(1.8,1.5))
num_k = 100
log_cum = np.log10(1.0 - cum_ori[:,0:num_k])
plt.contourf(log_cum, levels = 25, cmap= 'rainbow')
#plt.legend()
plt.xticks(np.linspace(0,num_k, 3), np.linspace(0, num_k, 3))
plt.yticks(np.linspace(0,len(tvals) -1, 3), np.linspace(np.amin(tvals), np.amax(tvals), 3))
plt.xlabel(r'$k$',fontsize=myaxis_font)
plt.ylabel(r'$s$',fontsize=myaxis_font)
plt.title(r'$\log(1-C(k))$ Orientation',fontsize=myaxis_font)
cbar =plt.colorbar()
cbar.set_ticks([np.amin(log_cum),np.amax(log_cum)])
cbar.set_ticklabels([r'%0.1f'% np.amin(log_cum), r'%0.1f'% np.amax(log_cum)])
plt.tight_layout()
plt.savefig('ori_task_CK_contour.pdf')
plt.show()

plt.figure(figsize=(1.8,1.5))
plt.contourf(gen_err_phase, levels = 25, cmap= 'rainbow')
plt.plot(np.linspace(0,len(ptheory)-1, len(ptheory)), np.argmin(gen_err_phase,axis = 0)-0.4, color = 'black')
plt.xticks(np.linspace(0,len(ptheory)-1, 3), [1,50,100])
plt.yticks(np.linspace(0,len(tvals)-1, 3), np.linspace(np.amin(tvals), np.amax(tvals), 3))
plt.xlabel(r'$P$',fontsize=myaxis_font)
plt.ylabel(r'$s$',fontsize=myaxis_font)
plt.title(r'$E_g$ Phase',fontsize=myaxis_font)
cbar = plt.colorbar(fraction = 0.1, ticks = [np.amin(gen_err_phase), np.amax(gen_err_phase)])
cbar.ax.set_yticklabels([r'%0.1f'% np.amin(gen_err_phase), r'%0.1f'% np.amax(gen_err_phase)])
plt.tight_layout()
plt.savefig('phase_task_contour.pdf')
plt.show()


plt.figure(figsize=(1.8,1.5))
num_k = 100
log_cum = np.log10(1.0-cum_phase[:,0:num_k])
print(log_cum)
plt.contourf(log_cum, levels = 25, cmap= 'rainbow')
#plt.legend()
plt.xticks(np.linspace(0,num_k, 3), np.linspace(0, num_k, 3))
plt.yticks(np.linspace(0,len(tvals) -1, 3), np.linspace(np.amin(tvals), np.amax(tvals), 3))
plt.xlabel(r'$k$',fontsize=myaxis_font)
plt.ylabel(r'$s$',fontsize=myaxis_font)
plt.title(r'$\log(1-C(k))$ Phase',fontsize=myaxis_font)
#cbar =plt.colorbar(fraction = 0.1, ticks = [np.amin(cum_ori[0:num_k]), np.amax(cum_ori[0:num_k])])
cbar = plt.colorbar()
cbar.set_ticks([np.amin(log_cum),np.amax(log_cum)])
cbar.set_ticklabels([r'%0.1f'% np.amin(log_cum), r'%0.1f'% np.amax(log_cum)])
plt.tight_layout()
plt.savefig('phase_task_CK_contour.pdf')
plt.show()

plt.figure(figsize=(1.8,1.5))
plt.contourf(gen_err_slant, levels = 25, cmap = 'rainbow')
plt.plot(np.linspace(0,len(ptheory)-1, len(ptheory)), np.argmin(gen_err_slant,axis = 0)+0.4, color = 'black')
plt.xticks(np.linspace(0,len(ptheory)-1, 3), [1,50,100])
plt.yticks(np.linspace(0,len(tvals)-1, 3), np.linspace(np.amin(tvals), np.amax(tvals), 3))
plt.xlabel(r'$P$',fontsize=myaxis_font)
plt.ylabel(r'$s$',fontsize=myaxis_font)
plt.title(r'$E_g$ Hybrid',fontsize=myaxis_font)
cbar = plt.colorbar(fraction = 0.1, ticks = [np.amin(gen_err_slant), np.amax(gen_err_slant)])
cbar.ax.set_yticklabels([r'%0.1f'% np.amin(gen_err_slant), r'%0.1f'% np.amax(gen_err_slant)])
plt.tight_layout()
plt.savefig('hybrid_task_contour.pdf')
plt.show()


plt.figure(figsize=(1.8,1.5))
num_k = 100
log_cum = np.log10(1.0 - cum_slant[:,0:num_k])
plt.contourf(log_cum, levels = 25, cmap= 'rainbow')
#plt.legend()
plt.xticks(np.linspace(0,num_k, 3), np.linspace(0, num_k, 3))
plt.yticks(np.linspace(0,len(tvals) -1, 3), np.linspace(np.amin(tvals), np.amax(tvals), 3))
plt.xlabel(r'$k$',fontsize=myaxis_font)
plt.ylabel(r'$s$',fontsize=myaxis_font)
plt.title(r'$\log(1-C(k))$ Hybrid',fontsize=myaxis_font)
cbar =plt.colorbar()
cbar.set_ticks([np.amin(log_cum),np.amax(log_cum)])
cbar.set_ticklabels([r'%0.1f'% np.amin(log_cum), r'%0.1f'% np.amax(log_cum)])
plt.tight_layout()
plt.savefig('hybrid_task_CK_contour.pdf')
plt.show()


all_s = []
Eg_ori = []
Eg_both = []
errs_ori = []
errs_both = []
all_ck_ori = []
all_ck_both = []
x_list = []
y_list = []
labels = []
repeat_pvals = []
repeat_ptheory = []
repeat_linsp_ck = []
all_ck_avg = []
Eg_avg = []
errs_avg = []

pvals = np.logspace(0,2.5,15).astype('int')
ptheory = np.linspace(0,400,50)

t_task = 0.8
"""

for i,t in tqdm(enumerate(tvals)):
    K = t*K_simp + (1-t)*K_comp
    s, v = sort_eigh(K)
    K = K/s[0]
    s = s/s[0]
    all_s += [s]
    coeff_ori = 1/(K.shape[0]) * (v.T @ y_ori)**2
    coeff_both =  1/(K.shape[0]) *(v.T @ y_phase)**2
    Eg_ori += [power_law.mode_errs(ptheory, s, coeff_ori, lamb).sum(axis = 0)]
    Eg_both += [power_law.mode_errs(ptheory, s, coeff_slant, lamb).sum(axis = 0)]
    errs_ori += [kernel_regression_expt(K,y_ori,lamb,pvals)]
    errs_both += [kernel_regression_expt(K,y_slant,lamb,pvals)]
    y_list += [s[0:100]]
    x_list += [np.linspace(1,100,100)]
    labels += [r'$t = %0.1f$' % t]
    repeat_pvals += [pvals]
    repeat_ptheory += [ptheory]
    ck = np.cumsum(coeff_ori)/np.sum(coeff_ori)
    all_ck_ori += [ck]
    repeat_linsp_ck += [np.linspace(1,len(ck),len(ck))]
    ck = np.cumsum(coeff_both)/np.sum(coeff_both)
    all_ck_both += [ck]

    Eg_avg += [ t_task* Eg_ori[-1] + (1-t_task)*Eg_both[-1] ]
    errs_avg += [ t_task*errs_ori[-1] + (1-t_task) * errs_both[-1] ]
    all_ck_avg += [ np.cumsum(t_task* coeff_ori + (1-t_task)* coeff_both) / np.sum(t_task* coeff_ori + (1-t_task)* coeff_both) ]


plot_tool(x_list,y_list, labels, r'$k$', r'$\lambda_k$', 'Mixture Spectra', 'simple_complex_mix_spectra.pdf')
plot_tool(repeat_ptheory, Eg_ori, labels, r'$p$', r'$E_g$', 'Orientation Task', 'simple_complex_mix_orientation_task.pdf', x_expt = repeat_pvals, y_expt = errs_ori)
plot_tool(repeat_ptheory, Eg_both, labels, r'$p$', r'$E_g$', 'Orientation and Phase', 'simple_complex_mix_both_task.pdf', x_expt = repeat_pvals, y_expt = errs_both)
plot_tool(repeat_linsp_ck, all_ck_ori, labels, r'$k$', r'$C(k)$', 'Orientation Task', 'simple_complex_mix_Ck_ori_task.pdf', style = 'semilogx')
plot_tool(repeat_linsp_ck, all_ck_both, labels, r'$k$', r'$C(k)$', 'Orientation and Phase', 'simple_complex_mix_Ck_both_task.pdf', style = 'semilogx')

plot_tool(repeat_ptheory, Eg_avg, labels, r'$p$', r'$E_g$', 'Combined Tasks', 'simple_complex_mix_orientation_avg_task.pdf', x_expt = repeat_pvals, y_expt = errs_avg)
plot_tool(repeat_linsp_ck, all_ck_avg, labels, r'$k$', r'$C(k)$', 'Combined Tasks', 'simple_complex_mix_Ck_avg_task.pdf', style = 'semilogx')

"""


# possible expts: 1. vary nonlinearity, 2. vary threshold
"""
theta_stim = np.linspace(-math.pi/2,math.pi/2, P)
theta_pr = 2*math.pi*np.random.random_sample(N)

y_hard = target_fn_no_phase(theta_stim, 8)
y_easy = target_fn_no_phase(theta_stim, 2)

plt.figure(figsize=(2.4,2))

plt.plot(theta_stim,y_easy, color = 'black', label = 'Easy Task', linewidth=line_width)
plt.xlabel(r'$\theta$',fontsize=myaxis_font)
plt.ylabel(r'$y(\theta)$',fontsize=myaxis_font)
plt.title('Easy Task',fontsize=myaxis_font)
plt.xticks([-math.pi/2, 0, math.pi/2], [r'$-\pi/2$',r'$0$',r'$\pi/2$'])
plt.tight_layout()
plt.savefig('easy_task_visuals.pdf')
plt.show()

plt.figure(figsize=(2.4,2))

plt.plot(theta_stim,y_hard, color = 'black', label = 'Hard Task', linewidth=line_width)
plt.xlabel(r'$\theta$',fontsize=myaxis_font)
plt.ylabel(r'$y(\theta)$',fontsize=myaxis_font)
plt.title('Hard Task',fontsize=myaxis_font)
plt.xticks([-math.pi/2, 0, math.pi/2], [r'$-\pi/2$',r'$0$',r'$\pi/2$'])

plt.tight_layout()
plt.savefig('hard_task_visuals.pdf')
plt.show()





# different nonlinearities

avals = [0.2,1,2,3]
all_errs = []
all_Eg = []
all_spectra = []
all_ck = []

all_errs_hard = []
all_Eg_hard = []
all_ck_hard = []
labels = []
ck_linsp = []
repeat_ptheory = []
repeat_pvals = []
x_list = []
repeat_theta=  []
all_k = []

lamb = 50
for i,a in enumerate(avals):
    x_list += [np.linspace(1,100,100)]
    repeat_pvals += [pvals]
    repeat_ptheory += [ptheory]
    repeat_theta += [theta_stim]
    labels += [r'$q = %d$' % a ]
    R = gabor_features_no_phase(theta_stim, theta_pr, sigma, thresh, a = a)
    K = R.T @ R
    u,s,v = sort_svd(R)
    K = K/s[0]
    s = s/s[0]
    k_plot = K[:,int( len(theta_stim)/2) ]
    all_k += [k_plot / np.amax(k_plot)]
    all_spectra += [s[0:100]]
    coeffs = 1/P * (v.T @ y_easy)**2
    all_ck += [np.cumsum(coeffs)/np.sum(coeffs)]
    ck_linsp += [np.linspace(1, coeffs.shape[0], coeffs.shape[0])]

    coeffs_hard = 1/P * (v.T @ y_hard)**2
    all_ck_hard += [np.cumsum(coeffs_hard)/np.sum(coeffs_hard)]


    #K_cutoff = 400
    #s = s[0:K_cutoff]
    #coeffs = coeffs[0:K_cutoff]
    lamba = lamb * np.trace(K)
    modes = power_law.mode_errs(ptheory, s, coeffs, lamba)
    E_g = modes.sum(axis = 0)
    errs = kernel_regression_expt(K, y_easy, lamba, pvals)
    all_errs += [errs]
    all_Eg += [E_g]

    all_errs_hard += [kernel_regression_expt(K,y_hard,lamba,pvals)]
    all_Eg_hard += [power_law.mode_errs(ptheory, s, coeffs_hard, lamba).sum(axis = 0)]


plot_tool(repeat_ptheory, all_Eg, labels, r'$p$', r'$E_g$', 'Easy Task', 'generalization_vs_q_easy.pdf', x_expt = repeat_pvals, y_expt = all_errs)
plot_tool(repeat_ptheory, all_Eg_hard, labels, r'$p$', r'$E_g$', 'Hard Task', 'generalization_vs_q_hard.pdf', x_expt = repeat_pvals, y_expt = all_errs_hard)
plot_tool(x_list,all_spectra, labels, r'$k$', r'$\lambda_k$', 'Spectra', 'spectra_vs_q.pdf')
plot_tool(ck_linsp, all_ck, labels, r'$k$', r'$C(k)$', 'Easy Task', 'ck_vs_q_easy.pdf', style = 'semilogx')
plot_tool(ck_linsp, all_ck_hard, labels, r'$k$', r'$C(k)$', 'Hard Task', 'ck_vs_q_hard.pdf', style = 'semilogx')
plot_tool(repeat_theta, all_k, labels, r'$\theta$', r'$K(\theta)$', 'Kernels', 'kernel_vs_q.pdf', style = 'linear')

# vary threshold experiment
thresh_vals = [0,0.1,0.2]

zvals = np.linspace(-0.5,0.5,100)
plt.figure(figsize=(2.4,2))

for i,t in enumerate(thresh_vals):
    psi = (zvals - t) * (zvals > t)
    plt.plot(zvals, psi , label = r'$a = %0.1f$' % t, linewidth=line_width)
plt.xlabel(r'$z$', fontsize= myaxis_font)
plt.legend()
#plt.xticks([])
#plt.yticks([])
plt.ylabel(r'$\psi(z-a)$', fontsize=myaxis_font)
plt.title('Sparsifying Threshold', fontsize=myaxis_font)
plt.tight_layout()
plt.savefig('sparsifying_threshold.pdf')
plt.show()

s_ts = []
target_spec = []
all_ck = []
errs = []
Eg = []
labels = []
x_list= []
ck_linsp = []
repeat_ptheory = []
repeat_pvals = []
all_k_plot = []
repeat_theta = []
for i, t in enumerate(thresh_vals):
    repeat_pvals += [pvals]
    repeat_ptheory += [ptheory]
    repeat_theta  += [theta_stim]
    labels += [r'$a = %0.1f$' % t]
    x_list += [np.linspace(1,100,100)]
    R = gabor_features_no_phase(theta_stim, theta_pr, sigma, t)
    K = R.T @ R
    k_plot = K[:,int(len(theta_stim)/2)]
    all_k_plot += [k_plot/np.amax(k_plot)]
    s,v = sort_eigh(K)
    s = s/s[0]
    K = K/s[0]
    lambt = lamb * np.trace(K)
    s_ts += [s[0:100]]
    coeff = 1/K.shape[0] * (v.T @ y_easy)**2
    ck_linsp += [np.linspace(1,coeff.shape[0], coeff.shape[0])]
    all_ck += [np.cumsum(coeff)/np.sum(coeff)]
    errs += [kernel_regression_expt(K, y_easy, lambt, pvals)]
    Eg += [power_law.mode_errs(ptheory, s, coeff, lambt).sum(axis = 0)]

plot_tool(x_list, s_ts, labels, r'$k$', r'$\lambda_k$', 'Sparse Codes', 'spectra_vary_sparsity.pdf')
plot_tool(ck_linsp, all_ck, labels, r'$k$', r'$C(k)$', 'Sparse Codes', 'cumulative_power_vary_sparsity.pdf', style = 'semilogx')
plot_tool(repeat_ptheory, Eg, labels, r'$p$', r'$E_g$', 'Easy Task', 'generalization_vary_sparsity.pdf', x_expt = repeat_pvals, y_expt = errs)
plot_tool(repeat_theta, all_k_plot, labels, r'$\theta$', r'$K(\theta)$', 'Kernels', 'kernel_vs_threshold.pdf', style = 'linear')
"""
