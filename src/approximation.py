r"""Check the quality of the approximation."""

from numpy.random import rand, seed
from numpy import dot, logspace, log10, zeros
from numpy.linalg import norm
from sklearn.metrics.pairwise import rbf_kernel
from operalib import DecomposableKernel, RBFCurlFreeKernel, RBFDivFreeKernel

from scipy.sparse.linalg import eigsh
# from scipy.linalg import norm

from efficient_decomposable_gaussian import EfficientDecomposableGaussianORFF
from efficient_curlfree_gaussian import EfficientCurlFreeGaussianORFF
from efficient_divfree_gaussian import EfficientDivergenceFreeGaussianORFF

from matplotlib.pyplot import (errorbar, savefig, tight_layout, legend, xscale,
                               yscale, xlabel, ylabel)


def main():
    r"""Plot figure: Kernel approximation with ORFF."""
    d = 4
    p = 4
    Dmax = 100000
    T, R, RR = 10, 1000, 10

    # seed(0)
    A = rand(p, p)
    A = dot(A, A.T)
    A /= norm(A)
    K = DecomposableKernel(A, rbf_kernel, {'gamma': .5})
    sp1 = zeros((RR, T))

    seed(0)
    for i in range(RR):
        for j, D in enumerate(logspace(0, log10(Dmax), T)):
            for _ in range(R):
                X1 = 2 * rand(1, d) - 1
                X2 = 2 * rand(1, d) - 1

                phiX1 = EfficientDecomposableGaussianORFF(X1, A, .5, int(D))
                phiX2 = EfficientDecomposableGaussianORFF(X2, A, .5, int(D))
                Ktilde = dot(phiX1, phiX2.H)
                v = abs(eigsh(Ktilde - K(X1, X2), k=1, which='LM')[0])
                if v > sp1[i, j]:
                    sp1[i, j] = v

    K = RBFCurlFreeKernel(gamma=.5)
    sp2 = zeros((RR, T))
    seed(0)
    for i in range(RR):
        for j, D in enumerate(logspace(0, log10(Dmax), T)):
            for _ in range(R):
                X1 = 2 * rand(1, d) - 1
                X2 = 2 * rand(1, d) - 1

                phiX1 = EfficientCurlFreeGaussianORFF(X1, .5, int(D))
                phiX2 = EfficientCurlFreeGaussianORFF(X2, .5, int(D))
                Ktilde = dot(phiX1, phiX2.H)
                v = abs(eigsh(Ktilde - K(X1, X2), k=1, which='LM')[0])
                if v > sp2[i, j]:
                    sp2[i, j] = v

    K = RBFDivFreeKernel(gamma=.5)
    sp3 = zeros((RR, T))
    seed(0)
    for i in range(RR):
        for j, D in enumerate(logspace(0, log10(Dmax), T)):
            for _ in range(R):
                X1 = 2 * rand(1, d) - 1
                X2 = 2 * rand(1, d) - 1

                phiX1 = EfficientDivergenceFreeGaussianORFF(X1, .5, int(D))
                phiX2 = EfficientDivergenceFreeGaussianORFF(X2, .5, int(D))
                Ktilde = dot(phiX1, phiX2.H)
                v = abs(eigsh(Ktilde - K(X1, X2), k=1, which='LM')[0])
                if v > sp3[i, j]:
                    sp3[i, j] = v

    errorbar(logspace(0, log10(Dmax), T).astype(int),
             sp1.mean(axis=0), sp1.std(axis=0),
             label=r'Gaussian decomposable')
    errorbar(logspace(0, log10(Dmax), T).astype(int),
             sp2.mean(axis=0), sp2.std(axis=0),
             label=r'Gaussian curl-free')
    errorbar(logspace(0, log10(Dmax), T).astype(int),
             sp3.mean(axis=0), sp3.std(axis=0),
             label=r'Gaussian divergence-free')
    xscale('log')
    yscale('log')
    xlabel(r'$D$ (Number of features)')
    ylabel(r'$||\widetilde{K}-K||_{\infty}$ (Error)')
    legend()
    tight_layout()
    savefig('approximation.pgf', bbox_inches='tight')

if __name__ == "__main__":
    main()
