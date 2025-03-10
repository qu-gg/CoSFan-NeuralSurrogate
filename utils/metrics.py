"""
@file metrics.py

Holds a variety of metric computing functions for time-series forecasting models
"""
import scipy.stats as stats
import numpy as np


def mse(output, target, **kwargs):
    """ Gets the mean of the per-pixel MSE for the given length of timesteps used for training """
    full_pixel_mses = (output - target) ** 2
    sequence_pixel_mse = np.mean(full_pixel_mses, axis=(1, 2))
    return sequence_pixel_mse, np.mean(sequence_pixel_mse), np.std(sequence_pixel_mse)


def tcc(u, x, **kwargs):
    m, n, w = u.shape
    res = []
    for i in range(m):
        correlation_sum = 0
        count = 0
        for j in range(n):
            a = u[i, j, :]
            b = x[i, j, :]
            if (a == a[0]).all() or (b == b[0]).all():
                count += 1
                continue
            correlation_sum = correlation_sum + stats.pearsonr(a, b)[0]
        correlation_sum = correlation_sum / (n - count)
        res.append(correlation_sum)
    res = np.array(res)
    return res, np.mean(res), np.std(res)


def scc(u, x, **kwargs):
    m, n, w = u.shape
    res = []
    for i in range(m):
        correlation_sum = 0
        count = 0
        for j in range(w):
            a = u[i, :, j]
            b = x[i, :, j]
            if (a == a[0]).all() or (b == b[0]).all():
                count += 1
                continue
            correlation_sum = correlation_sum + stats.pearsonr(a, b)[0]
        correlation_sum = correlation_sum / (w - count)
        res.append(correlation_sum)
    res = np.array(res)
    return res, np.mean(res), np.std(res)
