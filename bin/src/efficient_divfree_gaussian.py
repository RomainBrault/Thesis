r"""Efficient implementation of the Gaussian divergence-free kernel."""

from time import time

from pympler.asizeof import asizeof

from numpy.random import rand, seed
from numpy.linalg import norm
from numpy import dot, zeros, logspace, log10, matrix, int, eye, float
from scipy.sparse.linalg import LinearOperator
from sklearn.kernel_approximation import RBFSampler
from matplotlib.pyplot import savefig, subplots, tight_layout


def _rebase(phiX, W, Wn):
    return (phiX.reshape((phiX.shape[0], 1, 1, phiX.shape[1])) *
            (eye(W.shape[1]).reshape(1, W.shape[1], W.shape[1], 1) * Wn -
            W * W.reshape(1, 1, W.shape[1], phiX.shape[1]) / Wn)).reshape(
        (-1, W.shape[1] * Wn.shape[3]))


def NaiveDivergenceFreeGaussianORFF(X, gamma=1.,
                                    D=100, eps=1e-5, random_state=0):
    r"""Return the Naive ORFF map associated with the data X.

    Parameters
    ----------
    X : {array-like}, shape = [n_samples, n_features]
        Samples.
    gamma : {float},
        Gamma parameter of the RBF kernel.
    D : {integer},
        Number of random features.
    eps : {float},
        Cutoff threshold for the singular values of A.
    random_state : {integer},
        Seed of the generator.

    Returns
    -------
    \tilde{\Phi}(X) : array
    """
    phi_s = RBFSampler(gamma=gamma, n_components=D,
                       random_state=random_state)

    phiX = _rebase(phi_s.fit_transform(X),
                   phi_s.random_weights_.reshape((1, -1, 1, D)),
                   norm(phi_s.random_weights_, axis=0).reshape((1, 1, 1, -1)))

    return matrix(phiX)


def EfficientDivergenceFreeGaussianORFF(X, gamma=1.,
                                        D=100, eps=1e-5, random_state=0):
    r"""Return the Efficient ORFF map associated with the data X.

    Parameters
    ----------
    X : {array-like}, shape = [n_samples, n_features]
        Samples.
    gamma : {float},
        Gamma parameter of the RBF kernel.
    D : {integer},
        Number of random features.
    eps : {float},
        Cutoff threshold for the singular values of A.
    random_state : {integer},
        Seed of the generator.

    Returns
    -------
    \tilde{\Phi}(X) : array
    """
    phi_s = RBFSampler(gamma=gamma, n_components=D,
                       random_state=random_state)
    phiX = phi_s.fit_transform(X)
    W = phi_s.random_weights_.reshape((1, -1, 1, phiX.shape[1]))
    Wn = norm(phi_s.random_weights_, axis=0).reshape((1, 1, 1, -1))
    return LinearOperator((phiX.shape[0] * X.shape[1],
                           phiX.shape[1] * X.shape[1]),
                          matvec=lambda b: dot(_rebase(phiX, W, Wn), b),
                          rmatvec=lambda r: dot(_rebase(phiX, W, Wn).T, r),
                          dtype=float)


def main():
    r"""Plot figure: Efficient decomposable gaussian ORFF."""
    N = 100  # Number of points
    dmax = 100  # Input dimension
    D = 100  # Number of random features

    seed(0)

    R, T = 10, 10
    time_Efficient, mem_Efficient = zeros((R, T, 2)), zeros((R, T))
    time_naive, mem_naive = zeros((R, T, 2)), zeros((R, T))

    for i, d in enumerate(logspace(0, log10(dmax), T)):
        X = rand(N, int(d))

        # Perform \Phi(X)^T \theta with Efficient implementation
        for j in range(R):
            start = time()
            phiX1 = EfficientDivergenceFreeGaussianORFF(X, D)
            time_Efficient[j, i, 0] = time() - start
            theta = rand(phiX1.shape[1], 1)
            start = time()
            phiX1 * theta
            time_Efficient[j, i, 1] = time() - start
            mem_Efficient[j, i] = asizeof(phiX1, code=True)

        # Perform \Phi(X)^T \theta with naive implementation
        for j in range(R):
            start = time()
            phiX2 = NaiveDivergenceFreeGaussianORFF(X, D)
            time_naive[j, i, 0] = time() - start
            theta = rand(phiX2.shape[1], 1)
            start = time()
            phiX2 * theta
            time_naive[j, i, 1] = time() - start
            mem_naive[j, i] = asizeof(phiX2, code=True)

    # Plot
    f, axes = subplots(1, 3, figsize=(10, 4), sharex=True, sharey=False)
    axes[0].errorbar(logspace(0, log10(dmax), T).astype(int),
                     time_Efficient[:, :, 0].mean(axis=0),
                     time_Efficient[:, :, 0].std(axis=0),
                     label='Efficient decomposable ORFF')
    axes[0].errorbar(logspace(0, log10(dmax), T).astype(int),
                     time_naive[:, :, 0].mean(axis=0),
                     time_naive[:, :, 0].std(axis=0),
                     label='Naive decomposable ORFF')
    axes[1].errorbar(logspace(0, log10(dmax), T).astype(int),
                     time_Efficient[:, :, 1].mean(axis=0),
                     time_Efficient[:, :, 1].std(axis=0),
                     label='Efficient decomposable ORFF')
    axes[1].errorbar(logspace(0, log10(dmax), T).astype(int),
                     time_naive[:, :, 1].mean(axis=0),
                     time_naive[:, :, 1].std(axis=0),
                     label='Naive decomposable ORFF')
    axes[2].errorbar(logspace(0, log10(dmax), T).astype(int),
                     mem_Efficient[:, :].mean(axis=0),
                     mem_Efficient[:, :].std(axis=0),
                     label='Efficient decomposable ORFF')
    axes[2].errorbar(logspace(0, log10(dmax), T).astype(int),
                     mem_naive[:, :].mean(axis=0),
                     mem_naive[:, :].std(axis=0),
                     label='Naive decomposable ORFF')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[0].set_xlabel(r'$p=\dim(\mathcal{Y})$')
    axes[1].set_xlabel(r'$p=\dim(\mathcal{Y})$')
    axes[2].set_xlabel(r'$p=\dim(\mathcal{Y})$')
    axes[0].set_ylabel(r'time (s)')
    axes[2].set_ylabel(r'memory (bytes)')
    axes[0].set_title(r'Preprocessing time')
    axes[1].set_title(r'$\widetilde{\Phi}(X)^T \theta$ computation time')
    axes[2].set_title(r'$\widetilde{\Phi}(X)^T$ required memory')
    axes[0].legend(loc=2)
    tight_layout()
    savefig('efficient_divfree_gaussian.pgf', bbox_inches='tight')

if __name__ == "__main__":
    main()
