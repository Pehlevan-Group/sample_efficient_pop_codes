import numpy as np
import matplotlib.pyplot as plt
import math
import power_law
import matplotlib as mpl
from cycler import cycler
from tqdm import tqdm

mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
myfont = 6
myaxis_font = 8
plt.rcParams.update({'font.size': myfont})
line_width = 1


def nonlin(z, act = 'tanh'):
    if act == 'rect':
        return (z<=1) * (z>=-1) * z + (z>1) - (z>1)
    return np.tanh(z)

def simulate_RNN(x_vals, w_i, W_r, act = 'tanh'):
    P = x_vals.shape[0]
    T = x_vals.shape[1]
    step = 2.0/T
    tau = 0.08
    time_ratio = step/tau
    #r = np.zeros( (P, T, w_i.shape[0]) )
    r = np.random.standard_normal((P,T,w_i.shape[0]) )
    for i in range(P):
        z = np.zeros(w_i.shape[0])
        for t, xt in enumerate(x_vals[i,:]):
            if t > 0:
                z = z + time_ratio * (-z + W_r @ r[i,t-1,:] + w_i * xt)
                r[i,t,:] = nonlin(z, act)
    return r

def kernel_expt(K, y, pvals, lamb, tvals, P, num_repeats = 25):


    errs = np.zeros((len(pvals), num_repeats))
    for i,p in enumerate(pvals):
        if p == np.amax(pvals):
            #plt.figure(figsize=(2.4,2))
            print("oop")
        for n in range(num_repeats):
            rand_i = np.random.randint(0,K.shape[0],p)
            Ki = K[rand_i,:]
            Kii = Ki[:,rand_i]
            yi = y[rand_i]
            #yhat = Ki.T @ np.linalg.solve(Kii + 1/K.shape[0]*lamb*np.eye(p),  yi)
            yhat = Ki.T @ np.linalg.solve(Kii + lamb*np.eye(p),  yi)
            errs[i,n] = np.mean( (yhat-y)**2 )
            if p == np.amax(pvals) and n < 6:
                yhatplot = yhat.reshape((P,T))
                yplot = y.reshape((P,T))
                #plt.plot(tvals, yhatplot[n,:],  color = 'C%d' % n )
                #plt.plot(tvals, yplot[n,:], '--', color = 'black')

        if p == np.amax(pvals):
            print("done")
            #plt.title(r'Function Approximation', fontsize=myaxis_font)
            #plt.xlabel(r'$t$', fontsize=myaxis_font)
            #plt.ylabel(r'$y(t), f(t)$', fontsize=myaxis_font)
            #plt.tight_layout()
            #plt.savefig('learned_functions.pdf')
            #plt.show()
    return np.mean(errs, axis = 1)

N = 4000
T  = 100

np.random.seed(0)

vecs = np.zeros((2,100))
theta = math.pi/3
vecs[0,:] = np.linspace(0,1,100) * np.cos(theta)
vecs[1,:] = np.linspace(0,1,100) * np.sin(theta)

"""
plt.figure(figsize=(2.4,2))
plt.plot(vecs[0,:], vecs[1,:])
plt.plot([np.cos(theta)], [np.sin(theta)], '*', color = 'C0')
plt.plot(np.linspace(-1,1,100), np.zeros(100), color = 'black')
plt.plot(np.zeros(100), np.linspace(-1,1, 100), color = 'black')
theta_vals = np.linspace(0,theta,100)
plt.plot(0.5*np.cos(theta_vals), 0.5*np.sin(theta_vals), color = 'black')
plt.text( 0.4*np.sqrt(3/2), 1/4.0, r'$\theta$', size = 2*myaxis_font)
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.title('Reach Direction Cue', fontsize=myaxis_font)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig('reach_cue.pdf')
plt.show()
"""


tvals = np.linspace(0,2,T)
w_i = np.random.standard_normal(N)

P = 50
theta_star = math.pi * np.random.random_sample(P) - math.pi/2

t_stim = 1
x = np.outer(theta_star, (tvals < t_stim) * np.ones(len(tvals)) )

t_resp = 1
tau_inv_vals = [0,1,2]
all_y = []
for i, ti in enumerate(tau_inv_vals):
    all_y += [np.outer(theta_star, np.exp(-ti * (tvals-t_resp) ) * (tvals >= t_resp) * np.ones(len(tvals)) )]


#gvals = [0.7, 1, 1.5, 3]
all_K = []
lamb = 1e-6

ptheory = np.linspace(0, 200, 50)
gvals = np.linspace(0.5, 2.5, 40)
Eg = np.zeros((len(all_y), len(gvals), len(ptheory)))
for i, g in enumerate(tqdm(gvals)):
    W_r = g * 1/np.sqrt(N) * np.random.standard_normal((N,N))
    r = simulate_RNN(x, w_i, W_r)
    R = r.reshape((r.shape[0]*r.shape[1],r.shape[2]))
    K = 1/np.sqrt(R.shape[1]) * R @ R.T
    s,v = np.linalg.eigh(K)
    indsort = np.argsort(s)[::-1]
    s = s[indsort]
    v = v[:,indsort]


    for j, y in enumerate(all_y):
        y_vector = y.reshape(P*T)
        y_vector = y_vector/np.sqrt( np.mean(y_vector**2) )
        coeffs = 1/s.shape[0] *(v.T @ y_vector)**2
        #N = s.shape[0]
        Eg[j,i,:] = power_law.mode_errs(ptheory, s, coeffs, lamb).sum(axis = 0)

for j in range(len(all_y)):
    plt.figure(figsize=(2.4,2))
    plt.contourf(Eg[j,:,:], levels = 100, cmap= 'rainbow')
    plt.plot(np.linspace(0,len(ptheory)-1, len(ptheory)), np.argmin(Eg[j,:,:],axis = 0), color = 'black')
    #plt.colorbar()
    cbar = plt.colorbar(fraction = 0.1, ticks = [np.amin(gen_err_phase), np.amax(gen_err_phase)])
    cbar.ax.set_yticklabels([r'%0.1f'% np.amin(gen_err_phase), r'%0.1f'% np.amax(gen_err_phase)])

    if j == 0:
        plt.title(r'$\tau = \infty$',fontsize=myaxis_font)
    else:
        plt.title(r'$\tau = %0.1f$' % (tau_inv_vals[j]**(-1)),fontsize=myaxis_font)
    plt.xticks(np.linspace(0,len(ptheory)-1, 5), np.linspace(np.amin(ptheory), np.amax(ptheory), 5))
    plt.yticks(np.linspace(0,len(gvals)-1, 5), np.linspace(np.amin(gvals), np.amax(gvals), 5))
    plt.xlabel(r'$p$',fontsize=myaxis_font)
    plt.ylabel(r'$g$', fontsize=myaxis_font)
    plt.tight_layout()
    plt.savefig('contour_tau_%d.pdf' % tau_inv_vals[j])
    plt.show()





"""
plt.figure(figsize=(2.4,2))
for i in range(5):
    plt.plot(tvals, x[i,:], linewidth = line_width)
plt.title('Input Sequence',fontsize=myaxis_font)
plt.xlabel(r'$t$', fontsize=myaxis_font)
plt.ylabel(r'$\theta(t)$', fontsize=myaxis_font)
plt.tight_layout()
plt.savefig('input_seq.pdf')
plt.show()

plt.figure(figsize=(2.4,2))
for i,taui in enumerate(tau_inv_vals):
    if taui == 0:
        plt.plot(tvals, all_y[i][4,:]/theta_star[4], linewidth = line_width, label = r'$\tau = \infty$')
    else:
        plt.plot(tvals, all_y[i][4,:]/theta_star[4], linewidth = line_width, label = r'$\tau = %0.1f$' % (1.0/taui) )

plt.title('Target Functions',fontsize=myaxis_font)
plt.xlabel(r'$t$', fontsize=myaxis_font)
plt.ylabel(r'$y(t) / \theta^\mu$', fontsize=myaxis_font)
plt.legend()
plt.tight_layout()
plt.savefig('target_seq.pdf')
plt.show()


plt.figure(figsize=(2.4,2))
plt.loglog(s/s[0], linewidth = line_width)
plt.title('Spectrum',fontsize=myaxis_font)
plt.xlabel(r'$k$', fontsize=myaxis_font)
plt.ylabel(r'$\lambda_k$', fontsize=myaxis_font)
plt.ylim([1e-10,1])
plt.tight_layout()
plt.savefig('spectrum.pdf')
plt.show()

v_plot = v.reshape((P, T, v.shape[1]))
plt.figure(figsize=(2.4,2))
for k in range(4):
    arr = v_plot[0,:,k]

    plt.plot(tvals, arr * np.sign(arr[2]) , label = 'k = %d' % k)
plt.title(r'Eigenmodes $k$',fontsize=myaxis_font)
plt.xlabel(r'$t$', fontsize=myaxis_font)
plt.ylabel(r'$\phi_k(\theta_0,t)$', fontsize=myaxis_font)
plt.legend()
plt.tight_layout()
plt.savefig('eigfns_rnn.pdf')
plt.show()

plt.figure(figsize=(2.4,2))
for k in range(4):
    arr = v_plot[k,:,0]
    plt.plot(tvals, arr * np.sign(arr[2])  , label = r'$\mu = %d$' % k)
plt.title(r'Eigenmodes $\theta^\mu$', fontsize=myaxis_font)
plt.xlabel(r'$t$', fontsize=myaxis_font)
plt.ylabel(r'$\phi_0(\theta^\mu,t)$', fontsize=myaxis_font)
plt.legend()
plt.tight_layout()
plt.savefig('eigfns_rnn_k0.pdf')
plt.show()


plt.figure(figsize=(2.4,2))
for i, y in enumerate(all_y):
    coeff = (v.T @ y.reshape(P*T))**2
    Ck = np.cumsum(coeff)/np.sum(coeff)
    if i == 0:
        plt.semilogx(np.linspace(1,len(Ck),len(Ck)), Ck, linewidth = line_width, label = r'$\tau = \infty$')
    else:
        plt.semilogx(np.linspace(1,len(Ck),len(Ck)), Ck, linewidth = line_width, label = r'$\tau = %0.2f$' % (1.0/tau_inv_vals[i]))
plt.title(r'Cumulative Power',fontsize=myaxis_font)
plt.xlabel(r'$k$', fontsize=myaxis_font)
plt.ylabel(r'$C(k)$', fontsize=myaxis_font)
plt.tight_layout()
plt.savefig('cumulative_power.pdf')
plt.show()
"""



pvals = np.logspace(0, 3, 20).astype('int')
ptheory = np.logspace(0, 3, 250)

all_theory = np.zeros((len(gvals), len(all_y), len(ptheory)))
all_expt = np.zeros((len(gvals), len(all_y), len(pvals)))

lamb = 1e-3

all_s = []
all_v = []
all_coeffs = []

for i, g in enumerate(gvals):
    K = all_K[i]

    s,v = np.linalg.eigh(K)
    indsort = np.argsort(s)[::-1]
    s = s[indsort]
    v = v[:,indsort]

    all_s += [s]
    all_v += [v]

    K = K/s[0]
    s = s/s[0]

    print("trace and sum of eigs equal?")
    print("K.trace: %0.4f | s.sum() = %0.4f" % (np.trace(K), s.sum()))

    coeffsi = []
    for j, y in enumerate(all_y):
        y_vector = y.reshape(P*T)
        y_vector = y_vector/np.sqrt( np.mean(y_vector**2) )

        errs = kernel_expt(K, y_vector, pvals, lamb, tvals, P)
        coeffs = 1/s.shape[0] *(v.T @ y_vector)**2
        #coeffs = coeffs/coeffs.sum()
        N = s.shape[0]
        theory = power_law.mode_errs(ptheory, s, coeffs, N *lamb).sum(axis = 0)

        all_theory[i,j,:] = theory
        all_expt[i,j,:] = errs
        coeffsi += [coeffs]

    all_coeffs += [coeffsi]

plt.figure(figsize=(2.4,2))
for i, g in enumerate(gvals):
    s = all_s[i]
    s = s/s[0]
    plt.loglog(np.linspace(1,len(s),len(s)), s, label = r'$g = %0.1f$' % g, color = 'C%d' % (i+3))
plt.title(r'Spectra',fontsize=myaxis_font)
plt.xlabel(r'$k$', fontsize=myaxis_font)
plt.ylabel(r'$\lambda_k$', fontsize=myaxis_font)
plt.ylim([1e-12,1])
plt.legend()
plt.tight_layout()
plt.savefig('spectra_gain.pdf')
plt.show()


coeffs0 = all_coeffs[0]
plt.figure(figsize=(2.4,2))

for j,cj in enumerate(coeffs0):
    if j == 0:
        label = r'$\tau = \infty$'
    else:
        label = r'$\tau = %0.1f$' % (1.0/tau_inv_vals[j])
    plt.semilogx(np.linspace(1,len(cj),len(cj)),np.cumsum(cj)/cj.sum(), label = label)
plt.title(r'Cumulative Power $g = %0.1f$' % gvals[0],fontsize=myaxis_font)
plt.xlabel(r'$k$', fontsize=myaxis_font)
plt.ylabel(r'$C(k)$', fontsize=myaxis_font)
plt.legend()
plt.tight_layout()
plt.savefig('cum_power_nonchaotic.pdf')
plt.show()


coeffs0 = all_coeffs[-2]
plt.figure(figsize=(2.4,2))

for j,cj in enumerate(coeffs0):
    if j == 0:
        label = r'$\tau = \infty$'
    else:
        label = r'$\tau = %0.1f$' % (1.0/tau_inv_vals[j])
    plt.semilogx(np.linspace(1,len(cj),len(cj)),np.cumsum(cj)/cj.sum(), label = label)
plt.title(r'Cumulative Power $g = %0.1f$' % gvals[-2],fontsize=myaxis_font)
plt.xlabel(r'$k$', fontsize=myaxis_font)
plt.ylabel(r'$C(k)$', fontsize=myaxis_font)
plt.legend()
plt.tight_layout()
plt.savefig('cum_power_chaotic.pdf')
plt.show()


coeffs0 = [all_coeffs[i][0] for i in range(len(gvals))]
plt.figure(figsize=(2.4,2))

for j,cj in enumerate(coeffs0):
    label = r'$g = %0.1f$' % gvals[j]
    plt.semilogx(np.linspace(1,len(cj),len(cj)), np.cumsum(cj)/cj.sum(), label = label, color = 'C%d' % (j+3))
plt.title(r'Cumulative Power $\tau = \infty$',fontsize=myaxis_font)
plt.xlabel(r'$k$', fontsize=myaxis_font)
plt.ylabel(r'$C(k)$', fontsize=myaxis_font)
plt.legend()
plt.tight_layout()
plt.savefig('cum_power_tau_infty.pdf')
plt.show()



v_chaos = all_v[-2].reshape((P,T,v.shape[1]))
v_non = all_v[0].reshape((P,T,v.shape[1]))

plt.figure(figsize=(2.4,2))
for k in range(4):
    arr = v_chaos[0,:,k]
    plt.plot(tvals, arr * np.sign(arr[2]) , label = 'k = %d' % k)
plt.title(r'Eigenmodes $g = %0.1f$' % gvals[-2],fontsize=myaxis_font)
plt.xlabel(r'$t$', fontsize=myaxis_font)
plt.ylabel(r'$\phi_k(\theta_0,t)$', fontsize=myaxis_font)
plt.legend()
plt.tight_layout()
plt.savefig('eigfns_rnn_chaotic.pdf')
plt.show()

plt.figure(figsize=(2.4,2))
for k in range(4):
    arr = v_non[0,:,k]
    plt.plot(tvals, arr * np.sign(arr[2]) , label = 'k = %d' % k)
plt.title(r'Low Dim. Eigenmodes $g = %0.1f$' % gvals[0],fontsize=myaxis_font)
plt.xlabel(r'$t$', fontsize=myaxis_font)
plt.ylabel(r'$\phi_k(\theta_0,t)$', fontsize=myaxis_font)
plt.legend()
plt.tight_layout()
plt.savefig('eigfns_rnn_non_chaotic.pdf')
plt.show()



plt.figure(figsize=(2.4,2))
for i, g in enumerate(gvals):
    plt.loglog(ptheory, all_theory[i,0,:], label = r'$g = %0.1f$' % g, color = 'C%d' % (i+3))
    plt.loglog(pvals, all_expt[i,0,:], 'o', markersize=2.5,  color = 'C%d' %(i+3),)
plt.title(r'Gain Dependence $\tau = \infty$',fontsize=myaxis_font)
plt.xlabel(r'$p$', fontsize=myaxis_font)
plt.ylabel(r'$E_g$', fontsize=myaxis_font)
plt.legend()
plt.tight_layout()
plt.savefig('gain_dependence.pdf')
plt.show()




plt.figure(figsize=(2.4,2))
for i, taui in enumerate(tau_inv_vals):
    if taui != 0:
        plt.loglog(ptheory, all_theory[0,i,:], label = r'$\tau = %0.1f$' % (1.0/taui), color = 'C%d' % i)
    else:
        plt.loglog(ptheory, all_theory[0,i,:], label = r'$\tau = \infty$', color = 'C%d' % i)
    plt.loglog(pvals, all_expt[0,i,:], 'o', markersize=2.5, color = 'C%d' % i)
plt.title(r'Low Dimensional $g = %0.1f$' % gvals[0],fontsize=myaxis_font)
plt.xlabel(r'$p$', fontsize=myaxis_font)
plt.ylabel(r'$E_g$', fontsize=myaxis_font)
plt.legend()
plt.tight_layout()
plt.savefig('tau_dependence_non_chaotic.pdf')
plt.show()

plt.figure(figsize=(2.4,2))
for i, taui in enumerate(tau_inv_vals):
    if taui != 0:
        plt.loglog(ptheory, all_theory[-2,i,:], label = r'$\tau = %0.1f$' % (1.0/taui), color = 'C%d' % i)
    else:
        plt.loglog(ptheory, all_theory[-2,i,:], label = r'$\tau = \infty$', color = 'C%d' % i)
    plt.loglog(pvals, all_expt[-2,i,:], 'o', markersize=2.5, color = 'C%d' % i)
plt.title(r'High Dimensional $g = %0.1f$' % gvals[-2],fontsize=myaxis_font)
plt.xlabel(r'$p$', fontsize=myaxis_font)
plt.ylabel(r'$E_g$', fontsize=myaxis_font)
plt.legend()
plt.tight_layout()
plt.savefig('tau_dependence_chaotic.pdf')
plt.show()
