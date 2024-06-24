"""
Project: Convex clustering with general Moreau envelope.

Projected Newton for proximal operator of l_1-norm.


@author: Chester Holtz
"""

import numpy as np
import scipy
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

# Geometric progression: step sizes for line search
beta=0.9
step_sizes = 1*np.power(beta,np.linspace(0,100,num=60))
step_sizes=np.array(step_sizes)

def prox_l1_newton(Dtilde, w0, gamma, max_iter, tol):
    """
    Compute the proximal operator of f(u) = ||Du||_1 using Newton on the dual problem.
    
    Parameters:
    Dtilde (numpy.ndarray): Matrix D.
    w0 (numpy.ndarray): Input vector.
    gamma (float): Regularization parameter.
    max_iter (int): Maximum number of iterations.
    tol (float): Tolerance for convergence.
    
    Returns:
    numpy.ndarray: The optimized vector w_opt.
    """
    # Initialize variables
    DtildeT = Dtilde.T
    y = w0.copy()
    b = Dtilde @ y
    A = Dtilde @ DtildeT
    A = A.tocsr()
    Aj = BCOO.from_scipy_sparse(A)

    n, p = Dtilde.shape
    x = np.asarray(np.linalg.pinv(A.todense()))@b
    
    px = lambda x: np.maximum(np.minimum(x, gamma), -gamma)
    jpx = lambda x: jnp.maximum(jnp.minimum(x, gamma), -gamma)
    grad = lambda x : A @ x - b
    fdual = lambda x : 0.5*x.T@A@x - x.T@b
    jfdual = lambda x : 0.5*x.T@Aj@x - x.T@b
    fprimal = lambda x : 0.5*np.linalg.norm(x - y)**2 + gamma*np.linalg.norm(Dtilde@x,ord=1)

    def _line_search(s, Ih, x, g):
        x = x.at[Ih].set(jpx(x[Ih] - s*g))
        f = jfdual(x)
        return f, x
    
    # Newton  iterations
    for k in range(max_iter):
        g = grad(x)
        I1 = np.logical_and(np.isclose(x, -gamma), (g > 1e-4))
        I2 = np.logical_and(np.isclose(x, gamma), (g <= -1e-4))
        I = np.logical_or(I1, I2)
        Ih = ~I
        Ihs = np.nonzero(Ih)[0]
        _AI = A[Ihs,:]
        AI = _AI[:, Ihs]
        gI = g[Ih]
        
        # Update w
        PA = np.asarray(np.linalg.pinv(AI.todense()))
        _x_new = (PA@gI).squeeze()
        
        # Update v
        Fks, _x_news = jax.vmap(_line_search,in_axes=(0,None,None,None))(step_sizes, Ih, jnp.array(x), jnp.array(_x_new))
        F_kamin = np.argmin(Fks)
        x_new = _x_news[F_kamin][Ih]
        
        # Check for convergence using gap
        primal = y - DtildeT@x
        dual_gap = gamma * np.sum(np.abs(Dtilde@primal)) - x.T@Dtilde@primal
        print(f'primal objective {fprimal(primal):.3f} dual objective {fdual(x):.3f} dual gap {dual_gap:.3f} dual feas {np.all(np.abs(x) <= gamma)} residual {np.linalg.norm(AI.todense()@_x_new - gI):.3f}')
        
        # Update variables
        x[Ih] = x_new

    print('=====')
    
    return y - DtildeT@x