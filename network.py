# -*- coding: utf-8 -*-
"""Feed Forward Network class."""

import numpy as np
import factory
from util import l1_norm, ELU, convergence_series
from util import dampening_oscillation_series, boundary_saturation_series


class Network(object):
    """A network object trained via neuroplasticity to model a target function."""

    def __init__(self, inp, out=1):
        inp = inp + 1  # Account for bias node.
        hid = np.ceil(np.sqrt(inp * out))
        self.struct = np.array([inp, hid, out], dtype=int)

        # Node-related variables.
        self.V = np.sum(self.struct, dtype=int)
        self.nodes = factory.node_vect(self.V)

        # Arc-related variables.
        self.adj_mat = factory.dag_adjacency_mat(self.V, inp, out)
        self.arcs = factory.weight_mat(self.adj_mat)
        self.infl = factory.influence_tensor(self.V)

        # Training variables.
        self.accuracy = np.full(3, np.NaN, dtype=float)
        self.hebb = factory.hebbian_update_tensor(self.adj_mat)
        self.stasis = 1

        # Internal state variables.
        self.activity = np.copy(self.nodes)
        self.setpoint = 0
        self.time = 0

    def update_stats(self):
        """Update internal state variables.

        Recalculate each node's activity using a triangular weighted running mean and
        update the setpoint using a running mean of network activity.
        """
        self.activity += 2 * (self.nodes - self.activity) / (self.time + 1)

        inp = self.struct[0]
        net_activity = np.sum(self.activity[inp:]) / (self.V - inp)
        self.setpoint += (net_activity - self.setpoint) / self.time

    def update_influence_tensor(self):
        """Update accuracy-based influence tensor.

        Recalculate each arc's influence on the accuracy of the model using a
        backward-iterative method and store it in the tensor along Axis 0 := time.
        """
        self.infl = np.roll(self.infl, -1, axis=0)
        self.infl[0][:-1, :-1] = 0

        arc_vals = self.arcs * self.nodes[:, np.newaxis]
        scale = l1_norm(arc_vals, axis=0)

        for i in np.flip(np.argwhere(scale > 0).flatten()):
            node_infl = l1_norm(self.infl[0][i, :])
            self.infl[0][:, i] = np.abs(arc_vals[:, i]) * node_infl / scale[i]

    def update_homeostatic_factor(self, rule):
        """Update homeostatic factor.

        Recalculate the homeostatic factor using a given rule in order to restore
        network activity to a given setpoint.
        """
        inp = self.struct[0]
        net_activity = np.sum(self.activity[inp:]) / (self.V - inp)

        self.stasis = rule(self.stasis, net_activity, self.setpoint)

    def update_hebbian_tensor(self, rule):
        """Update hebbian update tensor.

        Recalculate the hebbian update matrix using a given rule and store it in the
        hebbian update tensor along Axis 0 := time.
        """
        update_mat = rule(self.hebb, self.infl)

        self.hebb = np.roll(self.hebb, -1, axis=0)
        self.hebb[0] = update_mat

    def evaluate(self, state):
        """Predict the output of target function when evaluated on state.

        Recalculate the node vector by computing an activation function on the sum of
        inputs for each node using a forward-iterative method then return output.
        """
        inp = self.struct[0]
        self.nodes[1:inp] = state

        for i in range(inp, self.V):
            self.nodes[i] = ELU(np.dot(self.arcs[:, i], self.nodes))

        self.time += 1
        self.update_stats()

        out = self.struct[2]
        return self.nodes[-out:]

    def train(self, state, target):
        """Update parameters using labeled input to approximate target function."""
        actual = self.evaluate(state)
        self.update_influence_tensor()

        error = np.abs(actual - target)
        self.accuracy = np.roll(self.accuracy, 1)
        self.accuracy[0] = 1 - error if np.isfinite(error) else 1

        # First and second derivative of model accuracy w.r.t. time
        d_A = (3 * self.accuracy[0] - 4 * self.accuracy[1] + self.accuracy[2]) / 2
        d2_A = self.accuracy[0] - 2 * self.accuracy[1] + self.accuracy[2]

        if d_A < 0 and d2_A <= 0:
            self.update_hebbian_tensor(dampening_oscillation_series)
        else:
            self.update_hebbian_tensor(boundary_saturation_series)

        rand = np.random.standard_normal(self.arcs.shape) * self.infl[0, :, :-1]
        self.update_homeostatic_factor(convergence_series)
        self.arcs = (self.arcs + self.hebb[0] + rand) * self.stasis

        return self.accuracy[0]
