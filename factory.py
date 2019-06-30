# -*- coding: utf-8 -*-
"""Factory functions for initializing network components."""

import numpy as np


def node_vect(V):
    """Create a node vector of shape V."""
    vect = np.zeros(V, dtype=float)
    vect[0] = 1  # Set bias node to 1.
    return vect


def dag_adjacency_mat(V, inp, out):
    """Create an acyclic digraph adjacency matrix of shape (V, V).

    Initialize a boolean adjacency matrix encoding for the max number of arcs (excluding
    intra-layer arcs for inputs and outputs) that have a topological ordering.
    """

    def arcs_in_row(n: int):
        nonlocal V, inp, out
        return (V - n - 1) * (n < V - out) - (inp - n - 1) * (n < inp)

    num_arcs = np.array([arcs_in_row(n) for n in range(V)])
    mat = num_arcs[:, np.newaxis] > np.flip(np.arange(V))
    return mat


def weight_mat(adj_mat):
    """Create a weight matrix of shape (V, V).

    Initialize a float weight matrix formatted along a boolean adjacency matrix with
    elements randomly sampled from a standard normal distribution.
    """
    num_arcs = np.count_nonzero(adj_mat)
    weights = np.random.standard_normal(num_arcs)

    mat = np.zeros(adj_mat.shape, dtype=float)
    mat[adj_mat] = weights
    return mat


def influence_tensor(V):
    """Create an influence tensor of shape (2, V, V + 1).

    Initialize a float accuracy-based influence tensor comprised of influence matrices
    archived along Axis 0 representing time.
    """
    tensor = np.zeros((2, V, V + 1), dtype=float)
    tensor[:, -1, -1] = [1, 1]  # Set accuracy influence to 1.
    return tensor


def hebbian_update_tensor(adj_mat):
    """Create a hebbian update tensor of shape (2, V, V).

    Initialize a float hebbian update tensor comprised of hebbian update matrices
    (preserving topological ordering) archived along Axis 0 representing time.
    """
    update_mat = adj_mat.astype(float)
    tensor = np.tile(update_mat, (2, 1, 1))
    return tensor
