import numpy as np
import scipy.optimize as scpopt
from scipy.special import gamma, digamma, polygamma, loggamma
from scipy.stats import nbinom

def ll_nb(X, r, p):

    big_gamma_term = np.sum(loggamma(X + r) - loggamma(X+1))
    little_gamma_term = -X.size*loggamma(r)
    pr_term = X.size * r * np.log(1-p+1e-8) + np.sum(X) * np.log(p+1e-8)

    return big_gamma_term + little_gamma_term + pr_term






class NBEstimator(object):

    def __init__(self):
        self.r = None
        self.p = None

    def fit(self, X):
        pass

    def get_params(self):

        outputs = {}
        outputs['r'] = self.r
        outputs['p'] = self.p

        return outputs

    def compute_ll(self, X):
        return ll_nb(X, self.r, self.p)

    def summary(self, X):
        print('r =', self.r)
        print('p =', self.p)
        print('ll =', self.compute_ll(X))


class NBEstimatorMinimize(NBEstimator):

    def ll_nb(self, X, params):
        return ll_nb(X, params[0], params[1])

    def grad_ll_nb(self, X, params):
        grads = np.zeros((2,))
        grads[1] = -X.size*params[0] / (1 - params[1] + 1e-8) + X.sum() / (params[1] + 1e-8)
        grads[0] = np.sum(digamma(X + params[0])) - X.size * digamma(params[0])\
                   + X.size * np.log(1 - params[1] + 1e-8)
        return grads

    def hess_ll_nb(self, X, params):
        hess = np.zeros((2,2))
        hess[0,0] = np.sum(polygamma(1, X + params[0])) - X.size * polygamma(1, params[0])
        hess[0,1] = hess[1,0] = - X.size / (1 - params[1] + 1e-8)
        hess[1,1] = - X.size * params[0] / ((1 - params[1])**2 + 1e-8) - X.sum() / (params[1] ** 2 + 1e-8)
        return hess

    def fit(self, X, params0=None, method='trust-constr', options=None):

        fun = lambda params: -self.ll_nb(X, params)
        jac = lambda params: -self.grad_ll_nb(X, params)
        hess = lambda params: -self.hess_ll_nb(X, params)


        if options == {}:
            options = None

        mean_X = X.mean()
        var_X = X.var()
        if params0 is None:
            # MME initialization
            r0 = max(mean_X**2 / (var_X - mean_X), 1e-5)
            p0 = np.clip(mean_X / (r0 + mean_X), 1e-5, 1-1e-5)
            params0 = np.array([r0,p0])

        bounds = scpopt.Bounds(lb=np.array([1e-6,1e-6]), ub=np.array([np.inf,1.-1e-6]), keep_feasible=True)

        if method != 'L-BFGS-B':
            res = scpopt.minimize(fun,params0,method=method,jac=jac,hess=hess,bounds=bounds,options=options)
        else:
            res = scpopt.minimize(fun, params0, method=method, jac=jac, bounds=bounds, options=options)

        if not(res.success):
            print("Not a success")
            print(res.message)

        self.r = res.x[0]
        self.p = res.x[1]







