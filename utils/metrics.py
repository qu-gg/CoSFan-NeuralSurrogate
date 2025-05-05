"""
@file metrics.py

Holds a variety of metric computing functions for time-series forecasting models
"""
import numpy as np
import scipy.stats as stats

from skimage.filters import threshold_otsu


def mse(output, target, **kwargs):
    """ Gets the mean of the per-step MSE for the given length of timesteps used for training """
    full_pixel_mses = (output - target) ** 2
    sequence_pixel_mse = np.mean(full_pixel_mses, axis=(1, 2))
    return sequence_pixel_mse, np.mean(sequence_pixel_mse), np.std(sequence_pixel_mse)


def mae(output, target, **kwargs):
    """ Gets the mean of the per-step MAE for the given length of timesteps used for training """
    full_pixel_mses = np.abs(output - target)
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


def dcc(u, x):
    m, n, w = u.shape
    dice_cc = []

    for i in range(m):
        u_row = u[i, :, 50]
        x_row = x[i, :, 50]

        thresh_u = threshold_otsu(u_row)
        u_scar_idx = np.where(u_row >= thresh_u)[0]
        thresh_x = threshold_otsu(x_row)
        x_scar_idx = np.where(x_row >= thresh_x)[0]

        intersect = set(u_scar_idx) & set(x_scar_idx)
        dice_cc.append(2 * len(intersect) / float(len(set(u_scar_idx)) + len(set(x_scar_idx))))

    dice_cc = np.array(dice_cc)
    return dice_cc
