# -*- coding: UTF-8 -*-



import os
import sys
import codecs

if '--interactive' not in sys.argv[1:]:
    if sys.version_info[0] == 2:
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout, 'strict')
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr, 'strict')
    else:
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer, 'strict')

if '/usr/local/texlive/2016/texmf-dist/scripts/pythontex' and '/usr/local/texlive/2016/texmf-dist/scripts/pythontex' not in sys.path:
    sys.path.append('/usr/local/texlive/2016/texmf-dist/scripts/pythontex')    
from pythontex_utils import PythonTeXUtils
pytex = PythonTeXUtils()

pytex.docdir = os.getcwd()
if os.path.isdir('.'):
    os.chdir('.')
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())
else:
    if len(sys.argv) < 2 or sys.argv[1] != '--manual':
        sys.exit('Cannot find directory .')
if pytex.docdir not in sys.path:
    sys.path.append(pytex.docdir)



pytex.id = 'py_efficient_linop_default'
pytex.family = 'py'
pytex.session = 'efficient_linop'
pytex.restart = 'default'

pytex.command = 'code'
pytex.set_context('')
pytex.args = ''
pytex.instance = '0'
pytex.line = '984'

print('=>PYTHONTEX:STDOUT#0#code#')
sys.stderr.write('=>PYTHONTEX:STDERR#0#code#\n')
pytex.before()

r"""Example of efficient implementation of Gaussian decomposable ORFF."""

from time import time

from numpy.linalg import svd
from numpy.random import rand, seed
from numpy import (dot, diag, sqrt, kron, zeros,
                   logspace, log10, matrix, eye, int)
from scipy.sparse.linalg import LinearOperator
from sklearn.kernel_approximation import RBFSampler
from matplotlib.pyplot import savefig, subplots


pytex.after()
pytex.command = 'block'
pytex.set_context('')
pytex.args = ''
pytex.instance = '1'
pytex.line = '1019'

print('=>PYTHONTEX:STDOUT#1#block#')
sys.stderr.write('=>PYTHONTEX:STDERR#1#block#\n')
pytex.before()

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
    D : {integer}
        Number of random features.
    eps : {float}
        Cutoff threshold for the singular values of A.
    random_state : {integer}
        Seed of the generator.

    Returns
    -------
    \tilde{\Phi}(X) : array
    """
    # Decompose A=BB^\transpose
    u, s, v = svd(A, full_matrices=False, compute_uv=True)
    B = dot(diag(sqrt(s[s > eps])), v[s > eps, :])

    # Sample a RFF from the scalar Gaussian kernel
    phi_s = RBFSampler(gamma=gamma, n_components=D, random_state=random_state)
    phiX = phi_s.fit_transform(X)

    # Create the ORFF linear operator
    return matrix(kron(phiX, B))


pytex.after()
pytex.command = 'block'
pytex.set_context('')
pytex.args = ''
pytex.instance = '2'
pytex.line = '1091'

print('=>PYTHONTEX:STDOUT#2#block#')
sys.stderr.write('=>PYTHONTEX:STDERR#2#block#\n')
pytex.before()

def EfficientDecomposableGaussianORFF(X, A, gamma=1.,
                                      D=100, eps=1e-5, random_state=0):
    r"""Return the efficient ORFF map associated with the data X.

    Parameters
    ----------
    X : {array-like}, shape = [n_samples, n_features]
        Samples.
    A : {array-like}, shape = [n_targets, n_targets]
        Operator of the Decomposable kernel (positive semi-definite)
    gamma : {float},
        Gamma parameter of the RBF kernel.
    D : {integer}
        Number of random features.
    eps : {float}
        Cutoff threshold for the singular values of A.
    random_state : {integer}
        Seed of the generator.

    Returns
    -------
    \tilde{\Phi}(X) : Linear Operator, callable
    """
    # Decompose A=BB^\transpose
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
                                                B.T)))


pytex.after()
pytex.command = 'code'
pytex.set_context('')
pytex.args = ''
pytex.instance = '3'
pytex.line = '1356'

print('=>PYTHONTEX:STDOUT#3#code#')
sys.stderr.write('=>PYTHONTEX:STDERR#3#code#\n')
pytex.before()

sys.path.append('../src/')
import efficient_decomposable_gaussian

efficient_decomposable_gaussian.main()


pytex.after()
pytex.command = 'code'
pytex.set_context('')
pytex.args = ''
pytex.instance = '4'
pytex.line = '1385'

print('=>PYTHONTEX:STDOUT#4#code#')
sys.stderr.write('=>PYTHONTEX:STDERR#4#code#\n')
pytex.before()

sys.path.append('../src/')
import efficient_curlfree_gaussian

efficient_curlfree_gaussian.main()


pytex.after()
pytex.command = 'code'
pytex.set_context('')
pytex.args = ''
pytex.instance = '5'
pytex.line = '1415'

print('=>PYTHONTEX:STDOUT#5#code#')
sys.stderr.write('=>PYTHONTEX:STDERR#5#code#\n')
pytex.before()

sys.path.append('../src/')
import efficient_divfree_gaussian

efficient_divfree_gaussian.main()


pytex.after()


pytex.cleanup()
