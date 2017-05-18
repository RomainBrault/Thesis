r"""Efficient implementation of the Gaussian ORFF decomposable kernel."""

from time import time

from pympler.asizeof import asizeof

from numpy.linalg import svd
from numpy.random import rand, seed
from numpy import (dot, diag, sqrt, kron, zeros,
                   logspace, log10, matrix, eye, int, float)
from scipy.sparse.linalg import LinearOperator
from sklearn.kernel_approximation import RBFSampler
from matplotlib.pyplot import savefig, subplots, tight_layout


def NaiveDecomposableGaussianORFF(X, A, gamma=1.,
                                  D=100, eps=1e-5, random_state=0):
    r"""Return the Naive ORFF map associated with the data X.

    Parameters
    ----------
    X : {array-like}, shape = [n_samples, n_features]
        Samples.
    A : {array-like}, shape = [n_targets, n_targets]
        Operator of the Decomposable kernel (positive semi-definite)
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
    # Decompose A=BB^T
    u, s, v = svd(A, full_matrices=False, compute_uv=True)
    B = dot(diag(sqrt(s[s > eps])), v[s > eps, :])

    # Sample a RFF from the scalar Gaussian kernel
    phi_s = RBFSampler(gamma=gamma, n_components=D, random_state=random_state)
    phiX = phi_s.fit_transform(X)

    # Create the ORFF linear operator
    return matrix(kron(phiX, B))


def EfficientDecomposableGaussianORFF(X, A, gamma=1.,
                                      D=100, eps=1e-5, random_state=0):
    r"""Return the Efficient ORFF map associated with the data X.

    Parameters
    ----------
    X : {array-like}, shape = [n_samples, n_features]
        Samples.
    A : {array-like}, shape = [n_targets, n_targets]
        Operator of the Decomposable kernel (positive semi-definite)
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
    \tilde{\Phi}(X) : Linear Operator, callable
    """
    # Decompose A=BB^T
    u, s, v = svd(A, full_matrices=False, compute_uv=True)
    B = dot(diag(sqrt(s[s > eps])), v[s > eps, :])

    # Sample a RFF from the scalar Gaussian kernel
    phi_s = RBFSampler(gamma=gamma, n_components=D, random_state=random_state)
    phiX = phi_s.fit_transform(X)

    # Create the ORFF linear operator
    cshape = (D, B.shape[0])
    rshape = (X.shape[0], B.shape[1])
    return LinearOperator((phiX.shape[0] * B.shape[1], D * B.shape[0]),
                          matvec=lambda b: dot(phiX, dot(b.reshape(cshape),
                                               B)),
                          rmatvec=lambda r: dot(phiX.T, dot(r.reshape(rshape),
                                                B.T)),
                          dtype=float)


def main():
    r"""Plot figure: Efficient decomposable gaussian ORFF."""
    N = 100  # Number of points
    pmax = 100  # Maximum output dimension
    d = 20  # Input dimension
    D = 100  # Number of random features

    seed(0)
    X = rand(N, d)

    R, T = 10, 10
    time_Efficient, mem_Efficient = zeros((R, T, 2)), zeros((R, T))
    time_naive, mem_naive = zeros((R, T, 2)), zeros((R, T))

    for i, p in enumerate(logspace(0, log10(pmax), T)):
        A = rand(int(p), int(p))
        A = dot(A.T, A) + eye(int(p))

        # Perform \Phi(X)^T \theta with Efficient implementation
        for j in range(R):
            start = time()
            phiX1 = EfficientDecomposableGaussianORFF(X, A, D)
            time_Efficient[j, i, 0] = time() - start
            theta = rand(phiX1.shape[1], 1)
            start = time()
            phiX1 * theta
            time_Efficient[j, i, 1] = time() - start
            mem_Efficient[j, i] = asizeof(phiX1, code=True)

        # Perform \Phi(X)^T \theta with naive implementation
        for j in range(R):
            start = time()
            phiX2 = NaiveDecomposableGaussianORFF(X, A, D)
            time_naive[j, i, 0] = time() - start
            theta = rand(phiX2.shape[1], 1)
            start = time()
            phiX2 * theta
            time_naive[j, i, 1] = time() - start
            mem_naive[j, i] = asizeof(phiX2, code=True)

    # Plot
    f, axes = subplots(1, 3, figsize=(10, 4), sharex=True, sharey=False)
    axes[0].errorbar(logspace(0, log10(pmax), T).astype(int),
                     time_Efficient[:, :, 0].mean(axis=0),
                     time_Efficient[:, :, 0].std(axis=0),
                     label='Efficient decomposable ORFF')
    axes[0].errorbar(logspace(0, log10(pmax), T).astype(int),
                     time_naive[:, :, 0].mean(axis=0),
                     time_naive[:, :, 0].std(axis=0),
                     label='Naive decomposable ORFF')
    axes[1].errorbar(logspace(0, log10(pmax), T).astype(int),
                     time_Efficient[:, :, 1].mean(axis=0),
                     time_Efficient[:, :, 1].std(axis=0),
                     label='Efficient decomposable ORFF')
    axes[1].errorbar(logspace(0, log10(pmax), T).astype(int),
                     time_naive[:, :, 1].mean(axis=0),
                     time_naive[:, :, 1].std(axis=0),
                     label='Naive decomposable ORFF')
    axes[2].errorbar(logspace(0, log10(pmax), T).astype(int),
                     mem_Efficient[:, :].mean(axis=0),
                     mem_Efficient[:, :].std(axis=0),
                     label='Efficient decomposable ORFF')
    axes[2].errorbar(logspace(0, log10(pmax), T).astype(int),
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
    savefig('efficient_decomposable_gaussian.pgf', bbox_inches='tight')

if __name__ == "__main__":
    main()
