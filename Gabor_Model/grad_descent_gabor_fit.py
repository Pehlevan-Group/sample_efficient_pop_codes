import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize
import math
import sys

# x = [b (scale), q (polynomial order), a (sparsity), s (lengthscale)]
def psi(x, cos_theta):
    b,q,a,s = x
    ri = np.cosh(s**2 * cos_theta) / np.cosh(s**2) - a
    ri = np.nan_to_num(ri)
    power = ri**q
    power = np.nan_to_num(power)
    return b * np.maximum(np.zeros(len(theta_vals)), power)

def get_spec(x, *args):
    cos_vec = args[1]
    cos_features = args[3]
    k = get_K(x, *args)
    coeff = 2/len(cos_vec) * cos_features @ k
    return coeff**2

def get_K(x, *args):
    cos_mat = args[0]
    cos_vec = args[1]
    psi_mat = psi(x, cos_mat)
    psi_vec = psi(x, cos_vec)
    return 1/cos_vec.shape[0] * psi_mat @ psi_vec

def get_grad_K(x, *args):
    b,q,a,s = x
    Khat = get_K(x, args)
    grad_v = np.zeros(4)
    grad_v[0] = Khat/b

    cos_mat = args[0]
    cos_vec = args[1]
    psi_mat = psi(x, *cos_mat)
    psi_vec = psi(x, *cos_vec)
    dpsi_mat_dq = b * (psi_mat > 0) * psi_mat * np.log(np.cosh(s*cos_mat) / np.cosh(s) - a)
    dpsi_vec_dq = b * (psi_vec > 0) * psi_vec * np.log(np.cosh(s*cos_mat) / np.cosh(s) - a)

    grad_v[1] = 1/cos_vec.shape[0] * dpsi_mat_dq @ dpsi_vec_dq
    ## finish this later if search goes poorly
    return

# args : [cos_mat, cos_vec, Ktrue]
def loss(x, *args):
    Ktrue = args[2]
    Khat = get_K(x, *args)
    myl = 0.5 * np.mean( (Ktrue-Khat)**2 )
    print(myl)
    return myl
# get gradient of loss wrt x
def loss_grad(x, *args):
    Ktrue = args[2]
    Khat = get_K(x, *args)
    grad_K = get_grad_K(x, *args)
    return grad_K @ (Khat - K_true)

def loss_spectrum(x, *args):
    spec_true = args[4]
    spec_hat = get_spec(x, *args)
    myl = 0.5*np.mean((spec_true - spec_hat)**2/spec_true)
    print(myl)
    return myl


P = 100
theta_vals = np.linspace(-math.pi/2, math.pi/2, P)
cos_vec = np.cos(theta_vals)

cos_features = np.zeros((int(P/2), P))
for k in range(int(P/2)):
    cos_features[k,:] = np.cos(2*k*theta_vals)

cos_mat = np.cos( np.outer(np.ones(P), theta_vals) - np.outer(theta_vals, np.ones(P))  )
args = (cos_mat, cos_vec)
x_true = [2.5, 1.5, 0.1, 3]
K_true = get_K(x_true, *args)
plt.plot(theta_vals, K_true)
plt.show()

args = (cos_mat, cos_vec, K_true, cos_features)
spec_true = get_spec(x_true, *args)
plt.loglog(spec_true)
plt.show()


args = (cos_mat, cos_vec, K_true, cos_features, spec_true)
x0 = [2, 3, 1e-2, 2]
constrs = {sp.optimize.LinearConstraint( np.eye(4), lb = np.array([1e-4, 0.01, 1e-3, 1e-1]), ub = np.array([10, 4, 0.6, 10])  , keep_feasible = True) }
Bounds = {sp.optimize.Bounds(lb = np.array([1e-1, 0.01, 1e-3, 1e-1]), ub = np.array([25, 3, 0.6, 10]))}
#result = sp.optimize.minimize(loss, x0, method = 'Powell', args = args, bounds = Bounds, tol = 1e-12, options = {'maxiter': 2000, 'disp': True})
result = sp.optimize.minimize(loss_spectrum, x0, method = 'trust-constr', args = args, constraints = constrs, tol = 1e-12, options = {'maxiter': 5000, 'disp': True})
x = result.x
success = result.success
print(success)

myspec = get_spec(x, *args)
myspec = myspec/myspec[0] * spec_true[0]
plt.loglog( np.linspace(1,len(spec_true), len(spec_true)), spec_true)
plt.loglog( np.linspace(1,len(spec_true), len(spec_true)), myspec)
plt.savefig('spec_plot.pdf')
plt.show()


K_hat = get_K(x, *args)
K_hat *= 1/K_hat[int(P/2)] * K_true[int(P/2)]
print("GT, Fit")
print(x_true)
print(x)
plt.plot(theta_vals, K_true, label = r'GT $q,a,\sigma = %0.1f, %0.1f, %0.1f$' % (x_true[1], x_true[2], x_true[3] ) )
plt.plot(theta_vals, K_hat, '--', label = r'Fit $q,a,\sigma = %0.1f, %0.1f, %0.1f$' %  (x[1], x[2],x[3]) )
plt.xlabel(r'$\theta$', fontsize=20)
plt.ylabel(r'$K(\theta)$', fontsize=20)
plt.title(r'Synthetic Problem', fontsize=20)
plt.legend()
plt.tight_layout()
plt.savefig('synthetic_fit_P_%d.pdf' % P)
plt.show()
