# -*- coding: utf-8 -*-
"""Utility functions for computing essential mathematical operations."""

import numpy as np


def l1_norm(x, axis=None):
    """Compute the l1-norm of an array."""
    return np.linalg.norm(x, ord=1, axis=axis)


def ELU(z):
    """Compute the exponential linear unit element-wise."""
    return z if z > 0 else np.exp(z) - 1


def convergence_series(stasis, net_activity, setpoint, dt=0.8):
    """Generate a homeostatic factor (bounded on (0, inf)) from network activity.

    Compute the next homeostatic factor in a scaled-time series to converge the network
    activity time series to a set point.

    Args:
        stasis (float): Current homeostatic factor.
        net_activity (float): Average activity of non-input nodes.
        setpoint (float): Desired network activity.
        dt (float), optional: Differential equation approximation.

    Returns:
        float: New homeostatic factor always greater than zero.

    """
    scale = np.abs(setpoint) + np.abs(net_activity)
    mean = (setpoint - net_activity) / scale if scale != 0 else 0
    return stasis * (1 + mean * dt) if stasis > 0 else 1


def dampening_oscillation_series(hebb, infl, dt=0.8):
    """Generate a hebbian update matrix (bounded on [-1, 1]) from influence tensor.

    Compute the next hebbian update matrix element-wise along an oscillating scaled-time
    series that decays asymptotically to 0.

    Args:
        hebb (ndarray): Current hebbian update tensor.
        infl (ndarray): Influence tensor.
        dt (float), optional: Differential equation approximation.

    Returns:
        ndarray: New hebbian update matrix comprising of floats.

    """
    delta = 1 + dt * dt
    theta = np.arctan(dt)

    a = infl[0][:, :-1] / dt
    b = infl[-1][:, :-1] / dt

    update_mat = np.zeros(hebb[0].shape, dtype=float)
    cond = b == 0

    # Compute elements whose respective past influence is zero.
    A = 2 * np.cos(a[cond] * theta) * theta
    B = np.log(delta) * np.sin(a[cond] * theta)
    C = delta ** (-a[cond] / 2) / (2 * theta)
    update_mat[cond] = (A + B) * hebb[0][cond] * C

    # Compute elements whose respective past influence is nonzero.
    A = np.sin((a[~cond] + b[~cond]) * theta)
    B = delta ** (-b[~cond] / 2) * np.sin(a[~cond] * theta)
    C = delta ** (-a[~cond] / 2) / np.sin(b[~cond] * theta)
    update_mat[~cond] = (A * hebb[0][~cond] - B * hebb[-1][~cond]) * C

    return np.clip(update_mat, -1, 1)  # BUG: values may explode to infinity!


def boundary_saturation_series(hebb, infl, dt=0.8):
    """Generate a hebbian update matrix (bounded on [-1, 1]) from influence tensor.

    Compute the next hebbian update matrix element-wise along a scaled-time series that
    converges to 1 or -1 depending on the signature of each element.

    Args:
        hebb (ndarray): Current hebbian update tensor.
        infl (ndarray): Influence tensor.
        dt (float), optional: Differential equation approximation.

    Returns:
        ndarray: New hebbian update matrix comprising of floats.

    """
    a = infl[0][:, :-1] / dt

    A = (1 - dt) ** a
    B = 1 - A

    return A * hebb[0] + B * np.sign(hebb[0])
