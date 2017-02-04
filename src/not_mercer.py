import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_kernels


def phi(x, w, D):
    Z = np.dot(x, w)
    return np.hstack((np.cos(Z), np.sin(Z))) / np.sqrt(D)


def createColourbar(lwr, upr, fig, axes):
    cax = fig.add_axes([.92, 0.1, 0.01, 0.8])
    norm = matplotlib.colors.LogNorm(vmin=lwr, vmax=upr, clip=False)
    c = matplotlib.colorbar.ColorbarBase(cax, cmap=plt.get_cmap('rainbow'),
                                         norm=norm, label='D=')
    plt.title(r'$\widetilde{K}$')
    return c


def main():
    T = 25

    cm_subsection = np.linspace(0, 1, T + 1)
    colors = [matplotlib.cm.rainbow(x) for x in cm_subsection]

    d = 1
    N = 250

    np.random.seed(0)
    X = np.linspace(-1, 1, N).reshape((N, d))
    K = pairwise_kernels(X, metric='rbf', gamma=1. / (2. * .1 ** 2))

    c = np.random.randn(N, 2)
    A = .5 * np.eye(2) + .5 * np.ones((2, 2))

    plt.close()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    f, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    for k, D in enumerate(np.logspace(0, 4, T)):
        D = int(D)
        np.random.seed(0)

        w = np.random.randn(d, D) / .1
        phiX = phi(X, w, D)
        Kt = np.dot(phiX, phiX.T)

        pred = np.dot(np.dot(Kt, c), A)
        axes[0, 0].plot(X, pred[:, 0], c=colors[k], lw=.5, linestyle='-')
        axes[0, 0].set_ylabel(r'$y_1$')
        axes[0, 1].plot(X, pred[:, 1], c=colors[k], lw=.5, linestyle='-')
        axes[0, 1].set_ylabel(r'$y_2$')

        w = np.random.randn(d, D) / .1
        phiX = phi(X, w, D)
        Kt = np.dot(phiX, phiX.T)

        pred = np.dot(np.dot(Kt, c), A)
        axes[1, 0].plot(X, pred[:, 0], c=colors[k], lw=.5, linestyle='-')
        axes[1, 0].set_xlabel(r'$x$')
        axes[1, 0].set_ylabel(r'$y_1$')
        axes[1, 1].plot(X, pred[:, 1], c=colors[k], lw=.5, linestyle='-')
        axes[1, 1].set_xlabel(r'$x$')
        axes[1, 1].set_ylabel(r'$y_2$')

    axes[0, 0].plot(X, np.dot(np.dot(K, c), A)[:, 0], c='k', lw=.5, label='K')
    axes[0, 1].plot(X, np.dot(np.dot(K, c), A)[:, 1], c='k', lw=.5, label='K')
    axes[1, 0].plot(X, np.dot(np.dot(K, c), A)[:, 0], c='k', lw=.5, label='K')
    axes[1, 1].plot(X, np.dot(np.dot(K, c), A)[:, 1], c='k', lw=.5, label='K')

    axes[0, 0].set_title(r'$\widetilde{K}u \approx Ku$, realization 1', x=1.1)
    axes[1, 0].set_title(r'$\widetilde{K}u \approx Ku$, realization 2', x=1.1)

    for xx in axes.ravel():
        xx.legend(loc=4)

    createColourbar(1, D, f, axes)
    plt.savefig('not_Mercer.pgf', bbox_inches='tight')

if __name__ == "__main__":
    main()
