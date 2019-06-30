# Neuroplasticity
A neural network training algorithm implementing synaptic plasticity models.

The weights of the network are updated using hebbian and homeostatic-inspired mechanisms.

### Hebbian Mechanism
For hebbian learning, arcs are "strengthened" by updating weights in the direction that predicts increased reward. These update vectors change when the reward declines rapidly and depend on how much the weights influenced the outcome.

### Homeostatic Mechanism
For homeostatic training, weights are scaled by a factor in an attempt to stabilize network activity to a dynamic setpoint. Network activity is computed as the arithmetic average of the average activation of each node. The setpoint is then computed as the running mean of this network activity which converges since the homeostatic weight updates reinforces the network activity to previously predicted values.
