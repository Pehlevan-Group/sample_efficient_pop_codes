import numpy as np
import matplotlib.pyplot as plt
import math
# change proportion of neurons


# number of stimuli, neurons
p_tot = 500
N = 2000
sigma = 0
beta = 5
t=0.5
N_s = 10

# preferred orientations and phases
code = 2*math.pi * np.random.random_sample((N,2))
# grating stimuli
data = np.zeros((p_tot,2))
data[:,0] = math.pi * np.random.random_sample(p_tot)
np.random.seed(0)
data[:,1] = math.pi * np.random.random_sample(p_tot)

def responses(code,data, t, sigma = 3, sim_non = 'relu'):
    N = code.shape[0]
    p = data.shape[0]
    Ns = int(t*N)
    Nc = N - Ns

    R = np.zeros((N,p))
    theta_diffs = np.cos( np.outer(code[:,0], np.ones(p)) - np.outer(np.ones(N), data[:,0]) )
    phi_diffs   = np.cos( np.outer(code[:,1], np.ones(p)) - np.outer(np.ones(N), data[:,1]) )
    phi_sum     = np.cos( np.outer(code[:,1], np.ones(p)) + np.outer(np.ones(N), data[:,1]) )
    Z = 0.5 * np.exp( - sigma/2 * theta_diffs ) * phi_sum + 0.5 * np.exp( sigma/2*theta_diffs) * phi_diffs
    R[0:Ns,:] = Z[0:Ns,:]
    if sim_non == 'relu':
        R[0:Ns,:] = Z[0:Ns,:] * (Z[0:Ns,:]>0)

    R[Ns:N,:]=0.25*np.exp(-sigma * theta_diffs[Ns:N,:])+0.25*np.exp(sigma*theta_diffs[Ns:N,:])+0.5*np.exp(-sigma)*np.outer(np.cos(2*code[Ns:N,1]), np.ones(p))
    return R

def compute_kernel(data,beta,t):
    P = data.shape[0]
    K = np.zeros((P,P))
    print(t)
    for i in range(P):
        for j in range(P):
            cos_th = np.cos(data[i,0]-data[j,0])
            cos_ph = np.cos(data[i,1]-data[j,1])
            cos_ph_p = np.cos(data[i,1]+data[j,1])
            K[i,j] = (1-t)* np.cosh(cos_th) + 0.5*t*(np.exp(-cos_th)*cos_ph_p + np.exp(cos_th)*cos_ph)
            #K[i,j] = np.cosh(cos_th)
    return K

# which_var is 0 for theta, and 1 for phase
def kernel_exps(K, data, p, which_var, num_avg = 100):
    # orientation discrimination
    Y = np.zeros((data.shape[0],2))
    Y[:,0] = np.cos(data[:,which_var])
    Y[:,1] = np.sin(data[:,which_var])
    err = np.zeros(num_avg)
    lamb = 1e-1
    for n in range(num_avg):
        perm = np.random.permutation( np.arange(K.shape[0]) )
        randinds = perm[0:p]
        testinds = perm[p:K.shape[0]]
        y_tr = Y[randinds,:]
        Kt = K[randinds,:]
        Ktt = Kt[:,randinds]
        K_test = Kt[:,testinds]
        alpha = np.linalg.solve(Ktt+ lamb*np.eye(p), y_tr)
        yhat = K_test.T @ alpha
        err[n] = 2*np.mean( (yhat - Y[testinds,:])**2 )
    return np.mean(err), np.std(err)

def grad_desc(R, data, p, which_var, num_avg = 50):
    N = R.shape[0]
    w = np.zeros((N,2))
    Y = np.zeros((data.shape[0],2))
    Y[:,0] = np.cos(data[:,which_var])
    Y[:,1] = np.sin(data[:,which_var])
    errs = []
    lamb=1e-6
    T = 1000
    eta = 5e-3
    R = N**(-0.5) * R
    print("p = %d" % p)
    for n in range(num_avg):
        randinds = np.random.randint(0,data.shape[0], p)
        y_tr = Y[randinds,:]
        R_tr = R[:,randinds]
        for t in range(T):
            w += - eta * R_tr @ (R_tr.T @ w - y_tr) - eta * lamb*w

            if t % 250 == 1:
                tr_err = np.mean( (R_tr.T @ w - y_tr)**2 )
                print( np.log10(tr_err ) )
        if tr_err < 1:
            errs.append(  np.mean( (R.T @ w - Y)**2 ) )
    err = np.array(errs)
    return np.mean(err), np.std(err)

p = 4
tvals = np.linspace(0.01, 1, num = 10)
num_t = len(tvals)

pvals = np.linspace(1,10,10).astype('int')
ori_means = np.zeros((num_t, len(pvals)))
ori_stds  = np.zeros((num_t, len(pvals)))
ph_means  = np.zeros((num_t, len(pvals)))
ph_stds   = np.zeros((num_t, len(pvals)))

"""
betavals= [1,2,5,10]
for beta in betavals:
    K = compute_kernel(data,beta)
    s = np.linalg.eigvalsh(K)
    plt.loglog(np.sort(s)[::-1], label = r'$\beta = %d$' % beta)
plt.show()

d = np.zeros((100,2))
for beta in betavals:
    d[:,0] = np.linspace(-math.pi, math.pi, 100)
    K = compute_kernel(d,2)
    plt.plot(d[:,0], K[50,:])
plt.xlabel(r'$\theta$')
plt.ylabel(r'$K(\theta)$')
plt.show()
"""

for i, t in enumerate(tvals):
    #R = responses(code, data, t)
    #K = 1/R.shape[0] * R.T @ R
    K = compute_kernel(data,5,t)
    for j,p in enumerate(pvals):
        s,v = np.linalg.eigh(K)
        ind = np.argsort(s)[::-1]
        s = s[ind]
        v = v[:,ind]
        ori_means[i,j], ori_stds[i,j] = kernel_exps(K, data, p, which_var=0)
        ph_means[i,j], ph_stds[i,j] = kernel_exps(K, data, p, which_var=1)

    #plt.plot(pvals, ori_means[i,:])
    #plt.plot(pvals, ph_means[i,:])
    #plt.show()

plt.figure(figsize=(25,5))
plt.subplot(1,3,1, position = [0,0,5,5])
plt.plot(tvals, ph_means[:,1], label = r'$p = 2$')
plt.plot(tvals, ph_means[:,4], label = r'$p = 5$')
plt.plot(tvals, ph_means[:,-1], label = r'$p = 10$')
plt.xlabel(r'$t$', fontsize=20)
plt.ylabel(r'$E_{phase}$',fontsize=20)
plt.legend()
#plt.tight_layout()


plt.subplot(1,3,2)
plt.plot(tvals, ori_means[:,1], label = r'$p = 2$')
plt.plot(tvals, ori_means[:,4], label = r'$p = 5$')
plt.plot(tvals, ori_means[:,-1], label = r'$p = 10$')
plt.xlabel(r'$t$', fontsize=20)
plt.ylabel(r'$E_{ori}$',fontsize=20)
#plt.tight_layout()

plt.subplot(1,3,3)
plt.plot(tvals, (ori_means[:,1] + ph_means[:,1]), label = r'$p = 2$')
plt.plot(tvals, (ori_means[:,4] + ph_means[:,4]), label = r'$p = 5$')
plt.plot(tvals, (ori_means[:,-1] + ph_means[:,-1]), label = r'$p = 10$')
#plt.fill_between(tvals, ori_means - ori_stds, ori_means+ori_stds, color = 'C0')
#plt.plot(tvals, ph_means[:,-1], color = 'C1')
#plt.legend()
plt.xlabel(r'$t$', fontsize=20)
plt.ylabel(r'$E_{phase} + E_{ori}$',fontsize=20)
#plt.tight_layout()
plt.savefig('vary_complex_cells.pdf')
#plt.fill_between(tvals, ph_means - ph_stds, ph_means+ph_stds, color = 'C1')
plt.show()
