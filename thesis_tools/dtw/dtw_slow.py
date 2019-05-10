from typing import Callable

import numpy as np


def compute_path(cost: np.ndarray) -> np.ndarray:
    # From: https://github.com/pierre-rouanet/dtw/blob/32e710adeb27764c4958fed5c57b6ed24fdc9bdf/dtw/dtw.py#L100

    i, j = cost.shape
    i -= 2
    j -= 2
    p, q = [i], [j]

    while i > 0 or j > 0:
        k = np.argmin((cost[i, j], cost[i, j + 1], cost[i + 1, j]))
        if k == 0:
            i -= 1
            j -= 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
        p.append(i)
        q.append(j)

    return np.c_[p[::-1], q[::-1]]


def dynamic_time_warping(s, t, distance_fn: Callable = None, precomputed_distances: np.ndarray = None):
    """
    Vanilla implementation of Dynamic Time Warping.
    For info and original source: https://en.wikipedia.org/wiki/Dynamic_time_warping
    :param s: the first sequence to align, with shape (n x ...)
    :param t: the second sequence to align, with shape (m x ...)
    :param distance_fn: a distance function that returns the distance between two elements
    of the sequences (according to any metric)
    :param precomputed_distances: matrix of distances precomputed w/ distance_fn; note that this parameter
     and distance_fn are mutually exclusive
    :return: a tuple with the cost as a (n x m) matrix and the best path
    """

    s = np.asarray(s)
    t = np.asarray(t)
    n = s.shape[0]
    m = t.shape[0]
    assert n > 1, m > 1

    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if precomputed_distances is not None:
                dist = precomputed_distances[i-1, j-1]
            else:
                dist = distance_fn(s[i-1], t[j-1])
            cost[i, j] = dist + min(cost[i-1, j],
                                    cost[i, j-1],
                                    cost[i-1, j-1])
    path = compute_path(cost)
    return cost[1:, 1:], path


def _traceback(D):

    from numpy import array, zeros, full, argmin, inf, ndim
    from math import isinf

    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)


def _dtw_pierre_rouanet(x, y, dist, warp=1, w=np.inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """

    from numpy import array, zeros, full, argmin, inf, ndim
    from math import isinf

    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)
    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path


def _run_basic_test():

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    a = np.zeros(500)
    a[30:60] = np.sin(np.arange(30)*0.5) + 10
    a[140:170] = np.sin(np.arange(30)*0.5) + 5
    a[350:380] = np.sin(np.arange(30)*0.5) + 7

    off = 15
    b = np.zeros(550)
    b[off:a.size+off] = a
    b[200:] = 0.0
    b[450:480] = np.sin(np.arange(30)*0.5) + 7

    # dist, cost, acc, path = _dtw_pierre_rouanet(a, b, lambda x_, y_: np.abs(x_ - y_))
    cost_m, path_m = dynamic_time_warping(a, b, lambda x_, y_: np.abs(x_ - y_))

    fig, axes = plt.subplots(2, 2)

    axes[0, 0].set_title('Data')
    axes[0, 0].plot(a, label='a')
    axes[0, 0].plot(b, label='b')
    axes[0, 0].legend()

    axes[0, 1].set_title('Mine')
    axes[0, 1].imshow(cost_m, origin='lower', cmap=cm.gray, interpolation='nearest')
    axes[0, 1].plot(*path_m.T, 'w')

    # axes[1, 1].set_title('Other')
    # axes[1, 1].imshow(acc, origin='lower', cmap=cm.gray, interpolation='nearest')
    # axes[1, 1].plot(*path, 'w')

    axes[1, 0].set_title('Aligned')
    axes[1, 0].plot(a[path_m[:, 0]], label='a')
    axes[1, 0].plot(b[path_m[:, 1]], label='b')
    axes[1, 0].legend()
    plt.show()


def _run_test2():

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import dtw_c

    x = np.r_[0, 0, 0, 0, 1, 1, 2, 2, 3, 2, 1, 1, 0, 0, 0, 0]
    y = np.r_[0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 1, 1, 0, 0]

    all_dists = np.zeros((len(x), len(y)), dtype=np.float32)
    for i in range(len(x)):
        all_dists[i] = np.abs(x[i] - y)
    cost_c, path_c = dtw_c.dynamic_time_warping_c(all_dists)

    cost_m, path_m = dynamic_time_warping(x, y, lambda x_, y_: np.abs(x_ - y_))
    cost, path = _dtw_pierre_rouanet(x, y, lambda x_, y_: np.abs(x_ - y_))[2:4]

    fig, axes = plt.subplots(3, 1, sharex='col')

    axes[0].imshow(cost.T, origin='lower', cmap=cm.gray, interpolation='nearest')
    axes[0].plot(path[0], path[1], 'w')

    axes[1].imshow(cost_m.T, origin='lower', cmap=cm.gray, interpolation='nearest')
    axes[1].plot(*path_m.T, 'w')

    axes[2].imshow(cost_c.T, origin='lower', cmap=cm.gray, interpolation='nearest')
    axes[2].plot(*path_c.T, 'w')
    plt.show()


def _run_test3():

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from sklearn.metrics.pairwise import euclidean_distances

    x = np.c_[[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [4, 3], [2, 3], [1, 1], [2, 2], [0, 1]].T
    y = np.c_[[1, 0], [1, 1], [1, 1], [2, 1], [4, 3], [4, 3], [2, 3], [3, 1], [1, 2], [1, 0]].T

    x = x[:, np.newaxis]
    y = y[:, np.newaxis]

    cost_m, path_m = dynamic_time_warping(x, y, euclidean_distances)
    plt.imshow(cost_m.T, origin='lower', cmap=cm.gray, interpolation='nearest')
    plt.plot(*path_m.T, 'w')
    plt.show()


if __name__ == '__main__':
    _run_test2()
