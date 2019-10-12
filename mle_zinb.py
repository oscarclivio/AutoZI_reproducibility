import numpy as np
import scipy.optimize as scpopt
from scipy.special import gamma, digamma, polygamma, loggamma
from scipy.stats import nbinom
from scipy.special import logsumexp
from mle_nb import ll_nb, NBEstimator, NBEstimatorMinimize
import matplotlib.pyplot as plt

plt.ioff()

def numbers_zeros_nonzeros(X):
    return (X==0).sum(), (X>0).sum()

def log_eps(el, eps=1e-9):
    return np.log(el+eps)

def div_eps(num, dem, eps=1e-9):
    return num / (dem + eps)

def ll_zinb(X, r, p, pi):

    n0, n1 = numbers_zeros_nonzeros(X)
    zero_term = n0 * logsumexp([log_eps(pi), log_eps(1-pi) + r*log_eps(1-p)])
    non_zero_term = n1 * log_eps(1-pi)

    big_gamma_term = np.sum((loggamma(X + r) - loggamma(X + 1))*(X>0))
    little_gamma_term = -n1 * loggamma(r)
    pr_term = n1 * r * log_eps(1 - p) + np.sum(X) * log_eps(p)

    return zero_term + non_zero_term + big_gamma_term + little_gamma_term + pr_term

def grads_ll_zinb(X, params, eps=1e-9):

    n0, n1 = numbers_zeros_nonzeros(X)
    r, p, pi = tuple(params)

    grads = np.zeros((3,))
    # dL/dr :
    grads[0] = n0 * log_eps(1-p) * np.exp( log_eps(1-pi) + r*log_eps(1-p)\
                                           - logsumexp([log_eps(pi), log_eps(1-pi) + r*log_eps(1-p)])) \
                + np.sum(digamma(X + r) * (X>0)) - n1 * digamma(r) + n1 * log_eps(1 - p)
    # dL/dp :
    grads[1] = - n0 * r * np.exp( log_eps(1-pi) + (r-1)*log_eps(1-p) \
                                  - logsumexp([log_eps(pi), log_eps(1-pi) + r*log_eps(1-p)])) \
                + div_eps(X.sum(), p) - div_eps(n1 * r, 1-p)
    # dL/dpi :
    grads[2] = n0 * np.exp( log_eps(1 - np.exp(r * log_eps(1-p))) \
                            - logsumexp([log_eps(pi), log_eps(1-pi) + r*log_eps(1-p)])) \
                - div_eps(n1, 1-pi)


    return grads

def ll_zinb_params(X, params):

    return ll_zinb(X, params[0], params[1], params[2])

def hessapprox_ll_zinb(X, params, eps=1e-9):


    hessapprox = np.zeros((3,3))
    eps_dirs = []
    for i in range(3):
        eps_dirs_el = np.zeros((3,))
        eps_dirs_el[i] += eps
        eps_dirs.append(eps_dirs_el)

    for i in range(3):
        grads_i = (grads_ll_zinb(X, params + eps_dirs[i]) - grads_ll_zinb(X, params - eps_dirs[i])) / (2*eps)

        hessapprox[i,:] = grads_i

    return hessapprox

def hess_ll_zinb(X, params, eps=1e-9):

    n0, n1 = numbers_zeros_nonzeros(X)
    r, p, pi = tuple(params)

    hess = np.zeros((3,3))

    # A few common formulae
    # log [ (1-p)^r ]
    log_one_minus_p_r = r*log_eps(1-p)
    # log [ (1-p)^(r-1) ]
    log_one_minus_p_r_1 = (r-1)*log_eps(1-p)
    # log [ 1-p ]
    log_one_minus_p = log_eps(1 - p)
    # log [ pi + (1-pi)(1-p)^r ]
    denom_one_minus_pi = logsumexp([log_eps(pi), log_eps(1-pi) + r*log_eps(1-p)])
    # log [ pi / (1-pi) + (1-p)^r ]
    denom_pi_one_minus_pi = logsumexp([log_eps(pi) - log_eps(1 - pi),  r * log_eps(1 - p)])

    # # d²L/d*dr
    # d²L/dr²
    hess[0,0] = np.sum(polygamma(1, X + params[0]) * (X>0)) - n1 * polygamma(1, params[0]) \
                    + n0 * (log_one_minus_p ** 2 * np.exp(log_one_minus_p_r - denom_pi_one_minus_pi) \
                            - log_one_minus_p ** 2 * np.exp(2*(log_one_minus_p_r - denom_pi_one_minus_pi)))
    # d²L/dpdr
    hess[0,1] = - div_eps(n1,1-p) - n0*(np.exp(log_one_minus_p_r_1 - denom_pi_one_minus_pi) \
                                        + log_one_minus_p_r*np.exp(log_one_minus_p_r_1 - denom_pi_one_minus_pi)) \
                    + n0 * log_one_minus_p_r * np.exp(log_one_minus_p_r + log_one_minus_p_r_1 - 2*denom_pi_one_minus_pi)
    # d2L/dpidr
    hess[0,2] = -n0 * log_one_minus_p * np.exp(log_one_minus_p_r - 2*denom_one_minus_pi)

    # # d²L/d*dp
    # d²L/drdp
    hess[1,0] = hess[0,1]
    # d²L/dp²
    hess[1,1] = - n1 * np.exp(log_eps(r) - 2*log_one_minus_p) - div_eps(X.sum(), p**2) \
                - n0 * (-r*(r-1)*np.exp( (r-2)*log_one_minus_p-denom_pi_one_minus_pi ) +\
                        r**2 * np.exp(2*log_one_minus_p_r_1 - 2*denom_pi_one_minus_pi))
    # d²L/dpidp
    hess[1,2] = n0 * r * np.exp(log_one_minus_p_r_1 - 2*denom_one_minus_pi)

    # # d²L/d*dpi
    # d2L/drdpi
    hess[2,0] = hess[0,2]
    # d²L/dpdpi
    hess[2,1] = hess[1,2]
    # d²L/dpidpi
    hess[2,2] = -div_eps(n1, (1-pi)**2) - n0 * np.exp(2*(log_eps(1 - np.exp(log_one_minus_p_r)) - denom_one_minus_pi))


    return hess



class ZINBEstimator(object):

    def __init__(self):
        self.r = None
        self.p = None
        self.pi = None

    def fit(self, X):
        pass

    def get_params(self):

        outputs = {}
        outputs['r'] = self.r
        outputs['p'] = self.p
        outputs['pi'] = self.pi

        return outputs

    def compute_ll(self, X):
        return ll_zinb(X, self.r, self.p, self.pi)

    def summary(self, X):
        print('r =', self.r)
        print('p =', self.p)
        print('pi =', self.pi)
        print('ll =', self.compute_ll(X))



class ZINBEstimatorMinimize(ZINBEstimator):

    def ll_zinb(self, X, params):
        return ll_zinb(X, params[0], params[1], params[2])

    def grad_ll_zinb(self, X, params):
        return grads_ll_zinb(X, params)

    def hess_ll_zinb(self, X, params):
        return hess_ll_zinb(X, params)

    def fit(self, X, params0=None, method='trust-constr', options=None):

        fun = lambda params: -self.ll_zinb(X, params)
        jac = lambda params: -self.grad_ll_zinb(X, params)
        hess = lambda params: -self.hess_ll_zinb(X, params)


        if options == {}:
            options = None


        mean_X_nonzero = X[X>0].mean()
        var_X_nonzero = X[X>0].var()
        if params0 is None:
            if mean_X_nonzero < var_X_nonzero:
                # Pseudo-MME initialization
                # p0 and r0 as in https://rdrr.io/bioc/polyester/src/R/get_params.R)
                p0 = 1 - mean_X_nonzero / var_X_nonzero
                r0 = (1 - p0) * mean_X_nonzero / p0
                # pi0 using p(x=0) = pi0 + (1-pi0)(1-p0)^r0
                pi0 = max([1e-2, div_eps((X==0).mean()\
                                         - np.exp(r0 * log_eps(1-p0) ) , 1. - np.exp(r0 * log_eps(1-p0) ))])
            else:
                p0 = 0.5
                pi0 = 0.2
                r0 = 1.
            params0 = np.array([r0, p0, pi0])

        constraints = scpopt.LinearConstraint(A=np.eye(3), lb=np.array([1e-6,1e-6,1e-6]),\
                                              ub=np.array([np.inf,1.-1e-6, 1.-1e-6]), keep_feasible=True)

        if method != 'L-BFGS-B':
            res = scpopt.minimize(fun,params0,method=method,jac=jac,hess=hess,constraints=constraints, options=options)
        else:
            res = scpopt.minimize(fun, params0, method=method, jac=jac, constraints=constraints, options=options)

        if not(res.success):
            print("Not a success")
            print(res.message)

        self.r = res.x[0]
        self.p = res.x[1]
        self.pi = res.x[2]
