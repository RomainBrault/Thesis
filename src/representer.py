import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def phi(x, w, D):
    Z = np.dot(x, w)
    return np.hstack((np.cos(Z), np.sin(Z))) / np.sqrt(D)


def main():
    d = 1
    D = 50

    N = 50
    Nt = 200

    np.random.seed(0)
    x = 2 * np.random.rand(N, d) - 1
    y = np.sin(10 * x)
    y += .5 * np.random.randn(y.shape[0], y.shape[1]) + 2. * x ** 2

    xt = np.linspace(-1, 1, Nt).reshape((-1, 1))
    yt = np.sin(10 * xt) + 2. * xt ** 2
    yt += .5 * np.random.randn(yt.shape[0], yt.shape[1])

    sigma = .3
    w = np.random.randn(d, D) / sigma

    phiX = phi(x, w, D)
    phiXt = phi(xt, w, D)

    plt.close()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    f, axis = plt.subplots(4, 3, gridspec_kw={'width_ratios': [3, 3, 1.5]},
                           figsize=(16, 6), sharex='col', sharey='col')
    f.subplots_adjust(hspace=.25)
    formatter = matplotlib.ticker.ScalarFormatter()
    formatter.set_powerlimits((-3, 4))

    for k, lbda in enumerate([1e-2, 5e-6, 1e-10, 0]):
        ck = np.linalg.lstsq(np.dot(phiX, phiX.T) + lbda * np.eye(N),
                             y, rcond=-1)[0]
        c = np.linalg.lstsq(np.dot(phiX.T, phiX) + lbda * np.eye(2 * D),
                            np.dot(phiX.T, y), rcond=-1)[0]
        cc = np.sum((phi(x, w, D) * ck), axis=0)
        cr = (cc - c.ravel()) / np.linalg.norm(c) * 100
        err = np.array([np.linalg.norm(np.dot(phiXt, c) - yt) ** 2 / Nt,
                        np.linalg.norm(np.dot(np.dot(phiXt,
                                                     phiX.T),
                                              ck) - yt) ** 2 / Nt,
                        np.linalg.norm(np.dot(phiXt, cr)) ** 2 / Nt,
                        np.linalg.norm(cr)])

        lmin = -1.8
        lmax = 3.
        axis[k, 0].set_xlim([-1.5, 1])
        axis[k, 0].set_ylim([lmin, lmax])
        axis[k, 0].plot(xt, np.dot(phiXt, c),
                        label=r'$\widetilde{\Phi}^* \theta$')
        axis[k, 0].plot(xt, np.dot(np.dot(phiXt, phiX.T), ck),
                        label=r'$\widetilde{K}u$', linestyle='-.')
        axis[k, 0].scatter(x, y, c='r', marker='+', label='train', lw=2)
        axis[k, 0].scatter(xt, yt, c='k', marker='.', label='test')
        axis[k, 0].legend(loc=3)
        axis[k, 0].set_ylabel('y')
        if k == 3:
            axis[k, 0].set_xlabel('x')

        lmin = -1.8
        lmax = 3.
        pred = np.dot(phi(xt, w, D), cr)
        axis[k, 1].set_xlim([-1.5, 1])
        axis[k, 1].set_ylim([lmin, lmax])
        axis[k, 1].plot(xt, pred,
                        label=r'$\widetilde{\Phi}^* \theta^{\parallel}$')
        axis[k, 1].scatter(x, y, c='r', marker='+', label='train', lw=2)
        axis[k, 1].scatter(xt, yt, c='k', marker='.', label='test')
        axis[k, 1].legend(loc=3)
        if k == 3:
            axis[k, 1].set_xlabel('x')

        xs = np.arange(cr.size)
        axis[k, 2].barh(xs, np.abs(cr), edgecolor="none", log=True)
        axis[k, 2].set_ylabel(r'$j$')
        if k == 3:
            axis[k, 2].set_xlabel(
                r'$|\theta^{\parallel}_j|$, \% of relative error')
    plt.savefig('representer.pgf', bbox_inches='tight')

    return err

if __name__ == "__main__":
    main()
