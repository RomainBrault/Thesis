import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from time import time
from operalib import toy_data_quantile
from sklearn.metrics.pairwise import euclidean_distances
from operalib import toy_data_quantile


def phi(X, omegas, D):
    Z = tf.matmul(X, omegas)
    phiX = tf.concat([tf.cos(Z), tf.sin(Z)], 1) / np.sqrt(D)
    return phiX


def phigrad(X, omegas, D):
    Z = tf.matmul(X, omegas)
    Zc = tf.cos(Z)
    Zs = tf.sin(Z)
    phiX = tf.concat([Zc, Zs], 1) / np.sqrt(D)
    phiXg = tf.concat([-omegas * Zs, omegas * Zc], 1) / np.sqrt(D)
    return phiX, phiXg


def model(phiX, phit, theta):
    return tf.matmul(tf.matmul(phiX, theta), tf.transpose(phit))


def loss(theta, X, t, y, omegas1, omegas2, D1, D2, lbda1, lbda2):
    phiX = phi(X, omegas1, D1)
    phit, phitg = phigrad(t, omegas2, D2)

    gxp = tf.matmul(phiX, theta)
    pred = tf.matmul(gxp, tf.transpose(phit))

    pin = tf.reduce_mean(tf.reduce_mean(tf.where(tf.greater_equal(pred, y),
                                                 (pred - y) * tf.transpose(t),
                                                 (y - pred) *
                                                 (1 - tf.transpose(t))), 1))
    cross = tf.reduce_mean(tf.reduce_mean(tf.maximum(tf.matmul(gxp,
                                          tf.transpose(phitg)), 0), 1))
    reg = tf.nn.l2_loss(theta) / (D1 * D2)
    return pin + lbda1 * cross + lbda2 * reg


def main():
    np.random.seed(0)

    print("Creating dataset...")
    N = 2000
    Nt = 1000
    d = 1
    p = 1
    probs = np.linspace(0.05, 0.95, 5)  # Quantile levels of interest
    x_train, y_train, z_train = toy_data_quantile(N)
    x_test, y_test, z_test = toy_data_quantile(Nt, probs=probs)

    ctype = tf.float32

    lbda1 = 1
    lbda2 = 1e-5
    ts = 100
    D1 = 100
    D2 = 100
    sigma1 = .25
    tn = np.random.rand(ts)
    sigma2 = np.median(euclidean_distances(tn.reshape(-1, 1)))
    with tf.device('/gpu:0'):
        X = tf.placeholder(ctype, [N, d], name='input_batch')
        y = tf.placeholder(ctype, [N, p], name='input_batch')
        t = tf.placeholder(ctype, [ts, 1])

        omegas1 = tf.Variable(tf.random_normal([d, D1],
                                               mean=0, stddev=1 / sigma1,
                                               dtype=ctype), trainable=False)
        omegas2 = tf.Variable(tf.random_normal([d, D2],
                                               mean=0, stddev=1 / sigma2,
                                               dtype=ctype), trainable=False)
        theta = tf.Variable(tf.random_normal([2 * D1, 2 * D2],
                                             mean=0, stddev=1,
                                             dtype=ctype), trainable=True)
        ls = loss(theta, X, t, y, omegas1, omegas2, D1, D2, lbda1, lbda2)

        opt = tf.train.RMSPropOptimizer(.00075).minimize(ls)
        test_X = tf.placeholder(ctype, [Nt, d], name='input_batch')
        test_t = tf.placeholder(ctype, [None, 1], name='input_batch')
        phitest_X = phi(test_X, omegas1, D1)
        phitest_t, phigtest_t = phigrad(test_t, omegas2, D2)

    config = tf.ConfigProto(allow_soft_placement=True)
    start = time()
    tt5 = 1 - np.linspace(0.05, 0.95, 5).reshape(-1, 1)
    with tf.Session(config=config) as session:
        init = tf.global_variables_initializer()
        session.run(init)

        for i in range(0, 1000):
            session.run(opt,
                        feed_dict={X: np.asarray(x_train, dtype=np.float32),
                                   y: np.asarray(y_train.reshape(-1, 1),
                                                 dtype=np.float32),
                                   t: np.asarray(tn.reshape(-1, 1),
                                                 dtype=np.float32)})
            if i % 100 == 0:
                print(session.run(ls,
                                  feed_dict={X: np.asarray(x_train,
                                                           dtype=np.float32),
                                             y: np.asarray(y_train.reshape(-1,
                                                                           1),
                                                           dtype=np.float32),
                                             t: np.asarray(tn.reshape(-1, 1),
                                                           dtype=np.float32)}))
        pred_test = session.run(model(phitest_X, phitest_t, theta),
                                feed_dict={test_X:
                                           np.asarray(x_test,
                                                      dtype=np.float32),
                                           test_t:
                                           np.asarray(tt5, dtype=np.float32)})
    session.close()
    print(time() - start)
    print('done')

    plt.figure(figsize=(20, 10))
    plt.gca().set_prop_cycle(None)
    for i, q in enumerate(z_test):
        plt.plot(x_test, q, '-', label='true quantile at ' + str(probs[i]))
    plt.gca().set_prop_cycle(None)
    plt.plot(x_test, pred_test, '--', label='Continuous Quantile')
    plt.scatter(x_train.ravel(), y_train.ravel(), marker='.', c='k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
